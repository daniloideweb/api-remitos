import os
import re
import json
import requests
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from google import genai
from google.genai import types

# Inicialización de FastAPI
app = FastAPI(
    title="API Extracción de Remitos",
    description="Servicio de visión e IA para extraer datos estructurados de remitos y comprobantes de carga.",
    version="1.0.0"
)

# Habilitar CORS para consultas desde cualquier frontend o ERP
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar cliente de Google GenAI usando la variable de entorno GEMINI_API_KEY
client = genai.Client()

# Esquema para llamadas POST
class RemitoRequest(BaseModel):
    archivo_url: str = Field(..., description="URL pública o compartida de Google Drive del remito")


def obtener_drive_direct_url(url: str) -> str:
    """Extrae el ID de Google Drive y construye el enlace de descarga directa."""
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if not match:
        match = re.search(r"id=([a-zA-Z0-9_-]+)", url)
    
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url


def descargar_archivo_drive(url: str) -> tuple[bytes, str]:
    """Descarga el binario del archivo desde Drive y detecta su tipo MIME."""
    download_url = obtener_drive_direct_url(url)
    session = requests.Session()
    
    response = session.get(download_url, timeout=25)
    
    # Manejo de confirmación de descarga de archivos grandes en Drive
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            download_url = f"{download_url}&confirm={v}"
            response = session.get(download_url, timeout=25)
            break
            
    if response.status_code != 200:
        raise HTTPException(
            status_code=400, 
            detail=f"No se pudo descargar el archivo desde Google Drive (HTTP {response.status_code}). Verifique que el enlace sea público."
        )
        
    content_type = response.headers.get("Content-Type", "").lower()
    
    # Normalización de tipo MIME para Gemini
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        mime_type = "application/pdf"
    elif "png" in content_type or url.lower().endswith(".png"):
        mime_type = "image/png"
    elif "jpeg" in content_type or "jpg" in content_type or url.lower().endswith(".jpg") or url.lower().endswith(".jpeg"):
        mime_type = "image/jpeg"
    else:
        # Por defecto tratarlo como imagen JPEG si no se especifica
        mime_type = "image/jpeg"
        
    return response.content, mime_type


def analizar_remito_con_gemini(file_bytes: bytes, mime_type: str) -> dict:
    """Envía el archivo a Gemini Flash con un prompt especializado en logística y transporte."""
    
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
    3. Devolver ÚNICAMENTE el bloque JSON, sin bloques markdown ```json ni texto adicional.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        
        texto_limpio = response.text.strip()
        return json.loads(texto_limpio)

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error en el procesamiento de visión/IA: {str(e)}"
        )


# Endpoint raíz para mantener activo el servicio (Cron-Job)
@app.get("/")
def home():
    return {
        "status": "online",
        "service": "API Remitos Raosa",
        "usage": "Consulte /procesar-remito vía GET con ?archivo_url=... o vía POST"
    }


# Llamado vía GET (Permite pegar la URL directo en el navegador como el zonificador)
@app.get("/procesar-remito")
@app.get("/procesar-remito/")
def procesar_remito_get(
    archivo_url: str = Query(..., description="URL compartida del archivo en Google Drive")
):
    file_bytes, mime_type = descargar_archivo_drive(archivo_url)
    resultado = analizar_remito_con_gemini(file_bytes, mime_type)
    return resultado


# Llamado vía POST (Para integración formal desde ERP o backend)
@app.post("/procesar-remito")
@app.post("/procesar-remito/")
def procesar_remito_post(payload: RemitoRequest):
    file_bytes, mime_type = descargar_archivo_drive(payload.archivo_url)
    resultado = analizar_remito_con_gemini(file_bytes, mime_type)
    return resultado
