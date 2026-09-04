import os
import re
import json
import time
import requests
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

app = FastAPI(
    title="API Extracción de Remitos",
    description="Servicio de visión e IA optimizado para digitalizar comprobantes de carga y remitos.",
    version="1.4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa cliente de Gemini utilizando GEMINI_API_KEY configurada en Render
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


def descargar_archivo_drive(url: str) -> tuple[bytes, str]:
    """Descarga el binario real omitiendo previsualizaciones HTML y optimizando la resolución a 1600px."""
    file_id = obtener_id_drive(url)
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    # Endpoint optimizado: 1600px de ancho para máxima velocidad de transferencia sin perder nitidez de lectura
    alt_url = f"https://lh3.googleusercontent.com/d/{file_id}=s1600"
    res = session.get(alt_url, headers=headers, timeout=25)

    # Si por alguna razón la imagen no responde por el endpoint optimizado, recurre a la descarga directa
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

    part_archivo = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
    part_texto = types.Part.from_text(text=prompt_instrucciones)
    contenido = types.Content(role="user", parts=[part_archivo, part_texto])

    # Configuración de alta velocidad: desactiva el thinking para extracción directa inmediata
    config_rapida = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.1,
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )

    # Configuración estándar de respaldo por si un modelo no admite thinking_budget=0
    config_estandar = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.1
    )

    modelos = ["gemini-2.5-flash", "gemini-3.6-flash"]
    ultimo_error = None

    for modelo in modelos:
        for intento in range(2):
            try:
                # Intento inicial con pensamiento desactivado para máxima velocidad
                response = client.models.generate_content(
                    model=modelo,
                    contents=[contenido],
                    config=config_rapida
                )
                return json.loads(response.text.strip())
            except Exception as e:
                ultimo_error = str(e)
                # Si el modelo rechaza thinking_budget=0, intenta con configuración estándar
                if "thinking_budget" in ultimo_error or "ThinkingConfig" in ultimo_error:
                    try:
                        response = client.models.generate_content(
                            model=modelo,
                            contents=[contenido],
                            config=config_estandar
                        )
                        return json.loads(response.text.strip())
                    except Exception as inner_e:
                        ultimo_error = str(inner_e)

                # Si es saturación momentánea (503), espera breve y reintenta
                if "503" in ultimo_error or "UNAVAILABLE" in ultimo_error:
                    time.sleep(1.5)
                    continue
                else:
                    break

    raise HTTPException(
        status_code=500,
        detail=f"Error en el procesamiento de visión/IA: {ultimo_error}"
    )


# Keep-alive para monitoreo
@app.get("/")
def home():
    return {"status": "online", "service": "API Remitos Raosa"}


# Llamado GET (Navegador y ERP)
@app.get("/procesar-remito")
@app.get("/procesar-remito/")
def procesar_remito_get(archivo_url: str = Query(..., description="URL de Drive del remito")):
    file_bytes, mime_type = descargar_archivo_drive(archivo_url)
    return procesar_con_gemini(file_bytes, mime_type)


# Llamado POST (Integración alternativa)
@app.post("/procesar-remito")
@app.post("/procesar-remito/")
def procesar_remito_post(payload: RemitoRequest):
    file_bytes, mime_type = descargar_archivo_drive(payload.archivo_url)
    return procesar_con_gemini(file_bytes, mime_type)
