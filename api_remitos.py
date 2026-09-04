import os
import re
import json
import requests
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

app = FastAPI(
    title="API Extracción de Remitos",
    description="Servicio de visión e inteligencia artificial para digitalizar comprobantes de carga y remitos.",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa cliente de Gemini con GEMINI_API_KEY del entorno
client = genai.Client()

class RemitoRequest(BaseModel):
    archivo_url: str = Field(..., description="URL de Google Drive del remito")


def obtener_id_drive(url: str) -> str:
    """Extrae el ID del archivo de cualquier enlace de Google Drive."""
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    match = re.search(r"id=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    return url


def descargar_archivo_drive(url: str) -> tuple[bytes, str]:
    """Descarga el binario real asegurando omitir HTMLs de previsualización."""
    file_id = obtener_id_drive(url)
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    # Primer intento: Endpoint directo de descarga
    download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&authuser=0"
    res = session.get(download_url, headers=headers, timeout=30)

    # Si Drive pide confirmación de virus/tamaño
    if "confirm=" in res.text:
        token = re.search(r"confirm=([0-9A-Za-z_]+)", res.text)
        if token:
            res = session.get(f"{download_url}&confirm={token.group(1)}", headers=headers, timeout=30)

    # Segundo intento si vino HTML: Endpoint de alta resolución de Google Photos/Drive
    if res.status_code != 200 or res.content.startswith(b"<!DOCTYPE html>") or b"<html" in res.content[:100].lower():
        alt_url = f"https://lh3.googleusercontent.com/d/{file_id}=s2500"
        res = session.get(alt_url, headers=headers, timeout=30)

    if res.status_code != 200 or res.content.startswith(b"<!DOCTYPE html>") or b"<html" in res.content[:100].lower():
        raise HTTPException(
            status_code=400,
            detail="No se pudo descargar la imagen o documento de Drive. Verifique que el archivo tenga acceso general en 'Cualquier persona con el enlace'."
        )

    # Detección de tipo MIME
    content_type = res.headers.get("Content-Type", "").lower()
    if res.content.startswith(b"%PDF") or "pdf" in content_type:
        mime_type = "application/pdf"
    elif res.content.startswith(b"\x89PNG") or "png" in content_type:
        mime_type = "image/png"
    else:
        mime_type = "image/jpeg"

    return res.content, mime_type


def procesar_con_gemini(file_bytes: bytes, mime_type: str) -> dict:
    prompt_instrucciones = """
    Sos un experto en logística de transporte y remitos de carga.
    Analizá detalladamente este comprobante y extraé los datos en un JSON estricto con las siguientes claves:

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
        "tipo": str o null,
        "numero": str o null,
        "fecha": str o null,
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

    Reglas:
    - Flete en destino o por cobrar -> condicion_pago = "DESTINO". Si está abonado/origen -> "ORIGEN".
    - Si figura contra reembolso o C/R -> contrarreembolso = true y poner el monto numérico en monto_contrarreembolso.
    - Responder ÚNICAMENTE el JSON crudo, sin etiquetas markdown de bloque.
    """

    try:
        # Construcción formal de partes para el SDK
        part_archivo = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
        part_texto = types.Part.from_text(text=prompt_instrucciones)

        contenido = types.Content(
            role="user",
            parts=[part_archivo, part_texto]
        )

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[contenido],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )

        return json.loads(response.text.strip())

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en el procesamiento de visión/IA: {str(e)}"
        )


@app.get("/")
def home():
    return {"status": "online", "service": "API Remitos Raosa"}


@app.get("/procesar-remito")
@app.get("/procesar-remito/")
def procesar_remito_get(archivo_url: str = Query(..., description="URL de Drive del remito")):
    file_bytes, mime_type = descargar_archivo_drive(archivo_url)
    return procesar_con_gemini(file_bytes, mime_type)


@app.post("/procesar-remito")
@app.post("/procesar-remito/")
def procesar_remito_post(payload: RemitoRequest):
    file_bytes, mime_type = descargar_archivo_drive(payload.archivo_url)
    return procesar_con_gemini(file_bytes, mime_type)
