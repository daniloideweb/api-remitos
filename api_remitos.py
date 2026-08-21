import os
import re
import time
import requests
from typing import Optional
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

app = FastAPI(
    title="API de Digitalización y Extracción de Remitos",
    description="Microservicio de extracción de comprobantes con detección de condición de pago (Origen/Destino) y contrarreembolso para ERP.",
    version="1.6.0"
)

# -----------------------------------------------------------------------------
# 1. MODELOS DE DATOS (PYDANTIC)
# -----------------------------------------------------------------------------
class RequestRemito(BaseModel):
    url_drive: str = Field(..., description="Enlace público del archivo alojado en Google Drive")

class UbicacionEntidad(BaseModel):
    razon_social: Optional[str] = Field(None, description="Razón Social o Nombre")
    cuit_dni: Optional[str] = Field(None, description="CUIT, CUIL o DNI")
    direccion: Optional[str] = Field(None, description="Calle y número")
    localidad: Optional[str] = Field(None, description="Localidad o Ciudad")
    provincia: Optional[str] = Field(None, description="Provincia")
    telefono: Optional[str] = Field(None, description="Teléfono de contacto")
    email: Optional[str] = Field(None, description="Correo electrónico de contacto")

class RespuestaExtraccion(BaseModel):
    tipo_comprobante: Optional[str] = Field(None, description="REMITO, FACTURA u OTRO")
    numero_remito: Optional[str] = Field(None, description="Número del comprobante / remito")
    remitente: UbicacionEntidad = Field(default_factory=UbicacionEntidad)
    destinatario: UbicacionEntidad = Field(default_factory=UbicacionEntidad)
    valor_declarado: Optional[float] = Field(None, description="Valor declarado/asegurado o subtotal neto sin impuestos.")
    condicion_pago_flete: Optional[str] = Field(None, description="ORIGEN (pago en origen, contado remitente, flete pago) o DESTINO (a pagar, cobro en destino, contra entrega de flete). Si no figura, null.")
    es_contrarreembolso: bool = Field(default=False, description="Indica si el comprobante especifica cobro de mercadería contra entrega / contrarreembolso / CR")
    monto_contrarreembolso: Optional[float] = Field(None, description="Monto a cobrar en destino por la mercadería si aplica contrarreembolso")
    bultos: Optional[int] = Field(None, description="Cantidad total de bultos")
    peso_kg: Optional[float] = Field(None, description="Peso total en kilogramos")

# -----------------------------------------------------------------------------
# 2. FUNCIONES DE DESCARGA Y PROCESAMIENTO
# -----------------------------------------------------------------------------
def extraer_file_id_drive(url: str) -> Optional[str]:
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    match_id = re.search(r"id=([a-zA-Z0-9_-]+)", url)
    if match_id:
        return match_id.group(1)
    return None

def descargar_bytes_drive(url: str) -> tuple[bytes, str]:
    file_id = extraer_file_id_drive(url)
    if not file_id:
        raise ValueError("No se pudo extraer el ID del archivo de Google Drive. Verificá que el enlace sea correcto.")
    
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(download_url, timeout=30)
    
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            download_url = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
            response = session.get(download_url, timeout=30)
            break
            
    if response.status_code != 200:
        raise ValueError(f"Error al descargar desde Drive (Código {response.status_code}). Asegurate de que el archivo tenga acceso general en 'Cualquier persona con el enlace'.")

    content_bytes = response.content
    content_type = response.headers.get("Content-Type", "")

    if content_bytes.startswith(b"%PDF"):
        mime_type = "application/pdf"
    elif content_bytes.startswith(b"\xff\xd8\xff"):
        mime_type = "image/jpeg"
    elif content_bytes.startswith(b"\x89PNG"):
        mime_type = "image/png"
    elif "image/jpeg" in content_type:
        mime_type = "image/jpeg"
    elif "image/png" in content_type:
        mime_type = "image/png"
    elif "pdf" in content_type:
        mime_type = "application/pdf"
    else:
        mime_type = "image/jpeg"

    return content_bytes, mime_type

def ejecutar_extraccion_gemini(archivo_bytes: bytes, mime_type: str) -> RespuestaExtraccion:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="No se encontró la GEMINI_API_KEY configurada en las variables de entorno.")

    prompt_analisis = """
    Sos un analista operativo experto en logística, transporte expreso y despacho de mercadería.
    Analizá el documento provisto (remito, guía, factura o comprobante de entrega) y extraé con precisión los siguientes datos:

    Reglas de Negocio Estrictas:
    1. Identificación de Entidades:
       - Remitente: Emisor que despacha (origen). Extraé CUIT/DNI, dirección, localidad, provincia, teléfono y correo electrónico.
       - Destinatario: Receptor final (destino). Extraé CUIT/DNI, dirección de entrega, localidad, provincia, teléfono y correo electrónico.
    2. Número de Comprobante: Extraé el número completo visible.
    3. Valor Declarado: Remito (valor declarado/asegurado de la mercadería) o Factura (subtotal neto gravado sin impuestos).
    4. Condición de Pago del Flete (condicion_pago_flete):
       - Si el flete o porte figura como pagado en origen, abono en origen, contado remitente, flete pago: asignar "ORIGEN".
       - Si el flete figura a pagar en destino, cobro en destino, flete a cobrar, cuenta corriente destino: asignar "DESTINO".
       - Si no se encuentra especificada la condición de pago del flete: asignar null.
    5. Condición de Contrarreembolso de Mercadería (CR):
       - Identificá si existe cobro contra entrega del valor de la mercadería, contrarreembolso, "C/R", "Cobrar al entregar".
       - es_contrarreembolso: Asignar true si está presente la condición, false de lo contrario.
       - monto_contrarreembolso: Extraé el importe numérico exacto a cobrar al destinatario por la mercadería (si aplica). Si no figura o es false, devolver null.
    6. Parámetros Físicos: Cantidad total de bultos y peso total en kg si figuran.
    7. Calidad: Si algún dato puntual no figura o es ilegible, devolvé null (o false para booleanos).
    """

    client = genai.Client(api_key=api_key)
    documento_part = types.Part.from_bytes(data=archivo_bytes, mime_type=mime_type)

    max_reintentos = 3
    for intento in range(max_reintentos):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=[documento_part, prompt_analisis],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RespuestaExtraccion
                ),
            )
            return RespuestaExtraccion.model_validate_json(response.text)
        except Exception as e:
            if intento < max_reintentos - 1:
                time.sleep(1.5 * (intento + 1))
            else:
                raise HTTPException(status_code=500, detail=f"Error durante el procesamiento del modelo: {str(e)}")

# -----------------------------------------------------------------------------
# 3. ENDPOINTS DISPONIBLES
# -----------------------------------------------------------------------------

# Endpoint GET (Llamadas directas por URL en navegador / ERP)
@app.get("/api/v1/procesar-remito-get", response_model=RespuestaExtraccion)
def procesar_remito_get(url_drive: str):
    try:
        archivo_bytes, mime_type = descargar_bytes_drive(url_drive)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo en la comunicación con Google Drive: {str(e)}")
    
    return ejecutar_extraccion_gemini(archivo_bytes, mime_type)

# Endpoint POST tradicional (JSON Body para ERP)
@app.post("/api/v1/procesar-remito", response_model=RespuestaExtraccion)
def procesar_remito_endpoint(payload: RequestRemito):
    try:
        archivo_bytes, mime_type = descargar_bytes_drive(payload.url_drive)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo en la comunicación con Google Drive: {str(e)}")
    
    return ejecutar_extraccion_gemini(archivo_bytes, mime_type)

# Endpoint Formulario (Para pruebas web)
@app.post("/api/v1/procesar-remito-form", response_model=RespuestaExtraccion)
def procesar_remito_form(url_drive: str = Form(...)):
    try:
        archivo_bytes, mime_type = descargar_bytes_drive(url_drive)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo en la comunicación con Google Drive: {str(e)}")
    
    return ejecutar_extraccion_gemini(archivo_bytes, mime_type)

# Panel visual interactivo de prueba en la raíz (/)
@app.get("/", response_class=HTMLResponse)
def panel_prueba_visual():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Digitalizador de Remitos - Raosa</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f1f5f9; padding: 40px; }
            .card { max-width: 650px; background: #fff; padding: 25px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: auto; }
            input[type="text"] { width: 100%; padding: 10px; margin: 10px 0; box-sizing: border-box; border: 1px solid #cbd5e1; border-radius: 4px; }
            button { width: 100%; background: #0284c7; color: white; border: none; padding: 12px; border-radius: 4px; font-weight: bold; cursor: pointer; }
            button:hover { background: #0369a1; }
            pre { background: #0f172a; color: #38bdf8; padding: 15px; border-radius: 6px; overflow-x: auto; display: none; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>Módulo de Pruebas - Transporte Raosa</h2>
            <form id="f">
                <label>Enlace de Google Drive:</label>
                <input type="text" id="u" placeholder="https://drive.google.com/file/d/..." required>
                <button type="submit">Procesar Documento</button>
            </form>
            <p id="c" style="display:none;color:#64748b;">Procesando comprobante con IA...</p>
            <pre id="r"></pre>
        </div>
        <script>
            document.getElementById('f').onsubmit = async (e) => {
                e.preventDefault();
                document.getElementById('c').style.display = 'block';
                document.getElementById('r').style.display = 'none';
                const fd = new URLSearchParams();
                fd.append('url_drive', document.getElementById('u').value);
                const res = await fetch('/api/v1/procesar-remito-form', { method: 'POST', body: fd });
                const data = await res.json();
                document.getElementById('c').style.display = 'none';
                document.getElementById('r').style.display = 'block';
                document.getElementById('r').textContent = JSON.stringify(data, null, 2);
            };
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_remitos:app", host="0.0.0.0", port=8000, reload=True)
