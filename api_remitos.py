import os
import re
import io
import json
import time
import requests
from PIL import Image, ImageOps
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

app = FastAPI(
    title="API Extracción de Remitos Raosa",
    description="Servicio de visión e IA optimizado para digitalizar comprobantes de carga y remitos.",
    version="2.7.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa cliente de Gemini con la variable de entorno GEMINI_API_KEY
client = genai.Client()

class RemitoRequest(BaseModel):
    archivo_url: str = Field(..., description="URL de Google Drive del remito")


def obtener_id_drive(url: str) -> str:
    """Extrae el ID del archivo de cualquier formato de enlace de Google Drive."""
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    match = re.search(r"id=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    return url


def normalizar_imagen(file_bytes: bytes, mime_type: str) -> tuple[bytes, str]:
    """Corrige la orientación física y metadatos EXIF de la imagen."""
    if not mime_type.startswith("image/"):
        return file_bytes, mime_type
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img = ImageOps.exif_transpose(img)
        buffer = io.BytesIO()
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue(), "image/jpeg"
    except Exception:
        return file_bytes, mime_type


def descargar_archivo_drive(url: str) -> tuple[bytes, str]:
    """Descarga el archivo real omitiendo previsualizaciones intermedias."""
    file_id = obtener_id_drive(url)
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    alt_url = f"https://lh3.googleusercontent.com/d/{file_id}=s1600"
    res = session.get(alt_url, headers=headers, timeout=25)

    if res.status_code != 200 or res.content.startswith(b"<!DOCTYPE html>") or b"<html" in res.content[:100].lower():
        download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&authuser=0"
        res = session.get(download_url, headers=headers, timeout=25)
        if "confirm=" in res.text:
            token = re.search(r"confirm=([0-9A-Za-z_]+)", res.text)
            if token:
                res = session.get(f"{download_url}&confirm={token.group(1)}", headers=headers, timeout=25)

    if res.status_code != 200 or res.content.startswith(b"<!DOCTYPE html>") or b"<html" in res.content[:100].lower():
        raise HTTPException(
            status_code=400,
            detail="No se pudo descargar la imagen o documento de Drive. Verifique que el archivo tenga acceso en 'Cualquier persona con el enlace'."
        )

    content_type = res.headers.get("Content-Type", "").lower()
    if res.content.startswith(b"%PDF") or "pdf" in content_type:
        mime_type = "application/pdf"
    elif res.content.startswith(b"\x89PNG") or "png" in content_type:
        mime_type = "image/png"
    else:
        mime_type = "image/jpeg"

    clean_bytes, clean_mime = normalizar_imagen(res.content, mime_type)
    return clean_bytes, clean_mime


def procesar_con_gemini(file_bytes: bytes, mime_type: str) -> dict:
    prompt_instrucciones = """
    Sos un experto liquidador y auditor documental de empresas de transporte de cargas y encomiendas.

    INSTRUCCIÓN DE LECTURA Y ORIENTACIÓN:
    La imagen puede estar invertida o rotada. Determina la orientación correcta antes de extraer los datos.

    REGLAS DE EXTRACCIÓN Y FORMATO:
    - Remitente y Destinatario: Extrae razón social, domicilio, localidad, provincia, teléfono y correo electrónico ('email') si figura.
    - CUIT / DNI: El campo 'cuit' debe extraerse estrictamente SIN guiones ni separadores. Si es una empresa u organismo, contendrá 11 dígitos juntos (ej: 30500810874); si es un particular y solo figura DNI, contendrá los 8 dígitos juntos.
    - Comprobante: Extrae el número de remito ('nro') unificado estrictamente en un solo bloque continuo de 12 dígitos sin guiones (4 para punto de venta y 8 para el número, rellenando con ceros a la izquierda si hace falta, ej: 000100047496) y el valor declarado ('val_decl').
    - Flete: Determina la condición de pago ('cond_pago') la cual debe ser obligatoriamente 'ORIGEN' (si paga el remitente) o 'DESTINO' (si paga el destinatario).
    - EXCLUYE COMPLETAMENTE: tipo de comprobante, fecha del remito, códigos postales, datos de carga, observaciones y total de flete.

    Estructura JSON estricta requerida:
    {
      "rem": {
        "rs": str o null,
        "cuit": str o null,
        "dom": str o null,
        "loc": str o null,
        "prov": str o null,
        "tel": str o null,
        "email": str o null
      },
      "dest": {
        "rs": str o null,
        "cuit": str o null,
        "dom": str o null,
        "loc": str o null,
        "prov": str o null,
        "tel": str o null,
        "email": str o null
      },
      "comprob": {
        "nro": str o null,
        "val_decl": float o null
      },
      "flete": {
        "cond_pago": "ORIGEN" o "DESTINO"
      }
    }

    Responde ÚNICAMENTE el JSON crudo respetando exactamente estas claves abreviadas, sin bloques markdown adicionales.
    """

    part_archivo = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
    part_texto = types.Part.from_text(text=prompt_instrucciones)
    contenido = types.Content(role="user", parts=[part_archivo, part_texto])

    config = types.GenerateContentConfig(
        response_mime_type="application/json"
    )

    max_intentos = 4
    ultimo_error = None

    for intento in range(max_intentos):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=[contenido],
                config=config
            )
            return json.loads(response.text.strip())
        except Exception as e:
            ultimo_error = str(e)
            
            if any(err in ultimo_error for err in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"]):
                espera = 4.0
                match_delay = re.search(r"retryDelay':\s*'(\d+)", ultimo_error)
                if match_delay:
                    espera = float(match_delay.group(1)) + 1.0
                else:
                    match_sec = re.search(r"retry in (\d+(?:\.\d+)?)s", ultimo_error)
                    if match_sec:
                        espera = float(match_sec.group(1)) + 1.0

                time.sleep(espera)
                continue

            break

    raise HTTPException(
        status_code=500,
        detail=f"Error en el procesamiento de visión/IA: {ultimo_error}"
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
