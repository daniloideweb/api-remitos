import os
import re
import json
import tempfile
import requests
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Inicialización de la aplicación FastAPI
app = FastAPI(
    title="API Extracción de Remitos",
    description="Servicio de visión e inteligencia artificial para digitalizar comprobantes de carga y remitos.",
    version="1.1.0"
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicialización del cliente de Google Gemini usando GEMINI_API_KEY del entorno
client = genai.Client()

# Esquema para llamadas POST
class RemitoRequest(BaseModel):
    archivo_url: str = Field(..., description="URL pública o compartida de Google Drive del remito")


def obtener_id_drive(url: str) -> str:
    """Extrae el identificador único del archivo de cualquier formato de enlace de Google Drive."""
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    match = re.search(r"id=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    return url


def descargar_y_procesar_remito(url: str) -> dict:
    """Descarga el archivo real omitiendo páginas intermedias de Drive y lo analiza con Gemini."""
    file_id = obtener_id_drive(url)
    
    # URL directa optimizada para evitar la pantalla de previsualización HTML
    direct_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&authuser=0"
    
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    response = session.get(direct_url, headers=headers, timeout=30)
    
    # Si Google Drive solicita confirmación de descarga para archivos pesados
    if "confirm=" in response.text:
        token_match = re.search(r"confirm=([0-9A-Za-z_]+)", response.text)
        if token_match:
            direct_url = f"{direct_url}&confirm={token_match.group(1)}"
            response = session.get(direct_url, headers=headers, timeout=30)

    # Si devolvió una página HTML o no pudo descargar el binario, recurre al endpoint de contenido de alta resolución
    if response.status_code != 200 or response.text.startswith("<!DOCTYPE html>") or "<html" in response.text[:100].lower():
        alt_url = f"https://lh3.googleusercontent.com/d/{file_id}=s2500"
        response = session.get(alt_url, headers=headers, timeout=30)
        
    if response.status_code != 200 or response.text.startswith("<!DOCTYPE html>") or "<html" in response.text[:100].lower():
        raise HTTPException(
            status_code=400,
            detail="No se pudo acceder al archivo de imagen o PDF en Google Drive. Verifique que el enlace esté configurado con acceso público ('Cualquier persona con el enlace')."
        )

    # Detección del tipo de archivo para el archivo temporal
    content_type = response.headers.get("Content-Type", "").lower()
    suffix = ".pdf" if "pdf" in content_type else ".jpg"

    # Almacenar temporalmente en disco para procesar vía File API nativo
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name

    prompt = """
    Sos un experto en logística de transporte de cargas y remitos. 
    Analizá detalladamente la imagen/documento provisto y extraé la información en un objeto JSON estricto con las siguientes claves:

    {
      "remitente": {
        "razon_social": str o null,
        "cuit": str o null,
        "domicilio": str o null,
        "localidad": str o null,
        "provincia": str o null
      },
      "destinatario": {
        "razon_social": str o null,
        "cuit": str o null,
        "domicilio": str o null,
        "localidad": str o null,
        "provincia": str o null
      },
      "comprobante": {
        "tipo": str o null (ej: "R", "X", "Guía"),
        "numero": str o null,
        "fecha": str o null (formato YYYY-MM-DD si es legible),
        "valor_declarado": float o null
      },
      "flete": {
        "condicion_pago": str ("ORIGEN" o "DESTINO" o null),
        "contrarreembolso": bool,
        "monto_contrarreembolso": float o null
      },
      "carga": {
        "cantidad_bultos": int o null,
        "peso_kg": float o null,
        "volumen_m3": float o null
      },
      "observaciones": str o null
    }

    Reglas críticas:
    1. Si figura pago en destino, cobrar en destino o similar, flete.condicion_pago debe ser "DESTINO". Si está pago o en origen, "ORIGEN".
    2. Si figura "C/R", "Contra Reembolso", "Cobrar al entregar" o un importe monetario explícito a cobrar al destinatario, contrarreembolso debe ser true y detallar monto_contrarreembolso.
    3. Devolver ÚNICAMENTE el bloque JSON, sin texto explicativo adicional.
    """

    uploaded_file = None
    try:
        # Subida segura al File API de Gemini
        uploaded_file = client.files.upload(file=tmp_path)

        response_gemini = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[uploaded_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        return json.loads(response_gemini.text.strip())

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en el procesamiento de visión/IA: {str(e)}"
        )
    finally:
        # Limpieza de recursos locales y remotos
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass


# Endpoint raíz de monitoreo Keep-Alive (Cron-job)
@app.get("/")
def home():
    return {
        "status": "online",
        "service": "API Remitos Raosa",
        "usage": "GET /procesar-remito?archivo_url=... o POST /procesar-remito con JSON body"
    }


# Llamado vía GET (Navegador directo como el zonificador)
@app.get("/procesar-remito")
@app.get("/procesar-remito/")
def procesar_remito_get(
    archivo_url: str = Query(..., description="URL compartida del archivo en Google Drive")
):
    return descargar_y_procesar_remito(archivo_url)


# Llamado vía POST (Integración ERP)
@app.post("/procesar-remito")
@app.post("/procesar-remito/")
def procesar_remito_post(payload: RemitoRequest):
    return descargar_y_procesar_remito(payload.archivo_url)
