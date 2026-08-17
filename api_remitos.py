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
    description="Microservicio para procesar enlaces de Google Drive y extraer datos logísticos para ERP.",
    version="1.0.0"
)

# -----------------------------------------------------------------------------
# 1. MODELOS DE DATOS (PYDANTIC)
# -----------------------------------------------------------------------------
class RequestRemito(BaseModel):
    url_drive: str = Field(..., description="Enlace público del archivo o escaneo alojado en Google Drive")

class UbicacionEntidad(BaseModel):
    razon_social: Optional[str] = Field(None, description="Razón Social o Nombre")
    cuit_dni: Optional[str] = Field(None, description="CUIT, CUIL o DNI")
    direccion: Optional[str] = Field(None, description="Calle y número")
    localidad: Optional[str] = Field(None, description="Localidad o Ciudad")
    provincia: Optional[str] = Field(None, description="Provincia")
    telefono: Optional[str] = Field(None, description="Teléfono de contacto si figura")
    email: Optional[str] = Field(None, description="Correo electrónico de contacto si figura")

class RespuestaExtraccion(BaseModel):
    tipo_comprobante: Optional[str] = Field(None, description="REMITO, FACTURA u OTRO")
    numero_remito: Optional[str] = Field(None, description="Número del comprobante / remito")
    remitente: UbicacionEntidad = Field(default_factory=UbicacionEntidad)
    destinatario: UbicacionEntidad = Field(default_factory=UbicacionEntidad)
    valor_declarado: Optional[float] = Field(None, description="Valor declarado asegurado o subtotal neto sin impuestos.")
    bultos: Optional[int] = Field(None, description="Cantidad total de bultos si figura")
    peso_kg: Optional[float] = Field(None, description="Peso total en kilogramos si figura")

# -----------------------------------------------------------------------------
# 2. FUNCIONES AUXILIARES DE DESCARGA DE GOOGLE DRIVE
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

# Lógica compartida de procesamiento de IA
def ejecutar_extraccion_gemini(archivo_bytes: bytes, mime_type: str) -> RespuestaExtraccion:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="No se encontró la GEMINI_API_KEY configurada.")

    prompt_analisis = """
    Sos un analista operativo experto en logística y despacho de mercadería.
    Analizá el documento provisto (remito, factura o comprobante de entrega) y extraé los campos exactos:
    Reglas de Negocio Estrictas:
    1. Identificación y Contacto:
       - Remitente: Emisor que despacha (origen). Extraé CUIT/DNI, dirección, localidad, provincia, teléfono y correo electrónico.
       - Destinatario: Receptor final (destino). Extraé CUIT/DNI, dirección de entrega, localidad, provincia, teléfono y correo electrónico.
    2. Número de Comprobante: Extraé el número completo.
    3. Valor Declarado: Remito (valor declarado/asegurado) o Factura (subtotal neto gravado sin impuestos).
    4. Calidad: Si algún dato numérico, geográfico o de contacto no figura o es ilegible, devolvé null.
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
                    response_schema=RespuestaExtraccion,
                    temperature=0.0,
                ),
            )
            return RespuestaExtraccion.model_validate_json(response.text)
        except Exception as e:
            if intento < max_reintentos - 1:
                time.sleep(2 * (intento + 1))
            else:
                raise HTTPException(status_code=500, detail=f"Error durante el procesamiento del modelo: {str(e)}")

# -----------------------------------------------------------------------------
# 3. ENDPOINTS DE LA API
# -----------------------------------------------------------------------------

# Endpoint JSON tradicional (para el ERP / sistemas)
@app.post("/api/v1/procesar-remito", response_model=RespuestaExtraccion)
def procesar_remito_endpoint(payload: RequestRemito):
    try:
        archivo_bytes, mime_type = descargar_bytes_drive(payload.url_drive)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo en la comunicación con Google Drive: {str(e)}")
    
    return ejecutar_extraccion_gemini(archivo_bytes, mime_type)


# Nuevo Endpoint de Formulario (con campo dedicado en la interfaz web de pruebas)
@app.post("/api/v1/procesar-remito-form", response_model=RespuestaExtraccion, include_in_schema=True)
def procesar_remito_form(url_drive: str = Form(..., description="Pegá aquí el enlace público de Google Drive del remito o factura")):
    try:
        archivo_bytes, mime_type = descargar_bytes_drive(url_drive)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo en la comunicación con Google Drive: {str(e)}")
    
    return ejecutar_extraccion_gemini(archivo_bytes, mime_type)


# Pantalla de pruebas visual amigable integrada en la raíz (/)
@app.get("/", response_class=HTMLResponse)
def panel_prueba_visual():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Prueba de Digitalización de Remitos - Raosa</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f4f6f9; margin: 0; padding: 40px; color: #333; }
            .container { max-width: 700px; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin: auto; }
            h2 { color: #1e293b; margin-top: 0; }
            label { font-weight: bold; display: block; margin-bottom: 8px; }
            input[type="text"] { width: 100%; padding: 12px; border: 1px solid #cbd5e1; border-radius: 6px; box-sizing: border-box; font-size: 14px; margin-bottom: 20px; }
            button { background-color: #2563eb; color: white; border: none; padding: 12px 20px; font-size: 16px; border-radius: 6px; cursor: pointer; width: 100%; font-weight: bold; }
            button:hover { background-color: #1d4ed8; }
            pre { background: #0f172a; color: #38bdf8; padding: 15px; border-radius: 6px; overflow-x: auto; font-size: 13px; margin-top: 20px; display: none; }
            .loading { color: #64748b; font-style: italic; display: none; margin-top: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Módulo de Pruebas - Transporte Raosa</h2>
            <p>Ingresá el enlace de Google Drive del documento para extraer los datos logísticos de forma automática.</p>
            <form id="remitoForm">
                <label for="url_drive">Enlace de Google Drive:</label>
                <input type="text" id="url_drive" name="url_drive" placeholder="https://drive.google.com/file/d/..." required>
                <button type="submit">Procesar Comprobante</button>
            </form>
            <div id="loading" class="loading">Procesando documento con inteligencia artificial... Aguarde unos segundos.</div>
            <pre id="resultado"></pre>
        </div>
        <script>
            document.getElementById('remitoForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const url = document.getElementById('url_drive').value;
                const resPre = document.getElementById('resultado');
                const loadingDiv = document.getElementById('loading');
                
                resPre.style.display = 'none';
                loadingDiv.style.display = 'block';

                const formData = new URLSearchParams();
                formData.append('url_drive', url);

                try {
                    const response = await fetch('/api/v1/procesar-remito-form', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                        body: formData
                    });
                    const data = await response.json();
                    loadingDiv.style.display = 'none';
                    resPre.style.display = 'block';
                    resPre.textContent = JSON.stringify(data, null, 2);
                } catch (err) {
                    loadingDiv.style.display = 'none';
                    resPre.style.display = 'block';
                    resPre.textContent = 'Error de conexión con el servidor.';
                }
            });
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_remitos:app", host="0.0.0.0", port=8000, reload=True)
