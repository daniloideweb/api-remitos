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
    title="API Extracción de Remitos",
    description="Servicio de visión e IA optimizado para digitalizar comprobantes de carga y remitos.",
    version="2.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa cliente de Gemini con GEMINI_API_KEY
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


def normalizar_imagen(file_bytes: bytes, mime_type: str) -> tuple[bytes, str]:
    """Corrige la orientación EXIF de la imagen si está rotada."""
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
    """Descarga el binario real de Google Drive."""
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
            detail="No se pudo descargar la imagen o documento de Drive. Verifique que el archivo tenga acceso público."
        )

    content_type = res.headers.get("Content-Type", "").lower()
    if res.content.startswith(b"%PDF") or "pdf" in content_type:
        mime_type = "application/pdf"
    elif res.content.startswith(b"\x89PNG") or "png" in content_type:
        mime_type = "image/png"
    else:
        mime_type = "image/jpeg"

    # Corrige orientación física de la imagen
    clean_bytes, clean_mime = normalizar_imagen(res.content, mime_type)
    return clean_bytes, clean_mime


def procesar_con_gemini(file_bytes: bytes, mime_type: str) -> dict:
    prompt_instrucciones = """
    Sos un experto liquidador y auditor documental de empresas de transporte de cargas y encomiendas.

    INSTRUCCIÓN DE LECTURA Y ORIENTACIÓN:
    La imagen puede estar invertida (de cabeza), rotada o inclinada. Antes de extraer los datos, determina la orientación correcta del comprobante para no confundir campos ni alucinar palabras.

    ESTRUCTURA DEL DOCUMENTO:
    El comprobante presenta comúnmente:
    1. Comprobante base de transporte (manuscrito o impreso con membrete, datos de Remitente, Destinatario, Localidad, Condición de Pago y Totales).
    2. En muchos casos, una etiqueta térmica adhesiva (con código de barras, número de guía/remito, bultos 'Bul:' y destino) pegada encima.

    REGLAS DE EXTRACCIÓN:
    - Etiqueta vs Manuscrito: Usa la etiqueta térmica como confirmación de lectura de número de guía, cantidad de bultos, destino y nombres si el manuscrito está tachado o poco legible.
    - Remitente y Destinatario: Identifica con total precisión quién despacha (REMITENTE) y quién recibe (DESTINATARIO). No intercambies sus roles.
    - Teléfonos y CUIT: Extrae números completos (teléfonos de contacto, CUIT o DNI manuscritos).
    - Flete y Pagos: 
      * Si dice 'ORIGEN' o pagado -> 'ORIGEN'. Si es por cobrar o 'DESTINO' -> 'DESTINO'.
      * Si figura contra reembolso o C/R -> contrarreembolso = true y coloca el importe monetario correspondiente.
    - Carga: Lee cuidadosamente la cantidad de bultos (ej. 22 bultos), descripción de la mercadería y peso si está indicado.

    Estructura JSON estricta requerida:
    {
      "remitente": {
        "razon_social": str o null,
        "cuit": str o null,
        "domicilio": str o null,
        "localidad": str o null,
        "provincia": str o null,
        "telefono": str o null
      },
      "destinatario": {
        "razon_social": str o null,
        "cuit": str o null,
        "domicilio": str o null,
        "localidad": str o null,
        "provincia": str o null,
        "codigo_postal": str o null,
        "telefono": str o null
      },
      "comprobante": {
        "tipo": str o null,
        "numero": str o null,
        "fecha": str o null,
        "valor_declarado": float o null
      },
      "flete": {
        "condicion_pago": str o null,
        "contrarreembolso": bool,
        "monto_contrarreembolso": float o null,
        "total_flete": float o null
      },
      "carga": {
        "cantidad_bultos": int o null,
        "peso_kg": float o null,
        "volumen_m3": float o null,
        "descripcion": str o null
      },
      "observaciones": str o null
    }

    Responde ÚNICAMENTE el JSON crudo sin bloques markdown.
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
            if any(err in ultimo_error for err in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"]):
                time.sleep(3 + (intento * 2))
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
