import os
import re
import io
import json
import time
import requests
from PIL import Image, ImageOps, ImageChops
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

app = FastAPI(
    title="API Extracción de Remitos y Facturas Raosa",
    description="Servicio de visión e IA optimizado para digitalizar comprobantes de carga y facturas con regla impositiva de subtotal.",
    version="2.14.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client()

# Cargar el listado oficial de localidades desde emi_loca.txt
def cargar_localidades() -> str:
    try:
        ruta_archivo = os.path.join(os.path.dirname(__file__), "emi_loca.txt")
        if os.path.exists(ruta_archivo):
            with open(ruta_archivo, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        pass
    return "00001,CORRIENTES CAPITAL,Corrientes"

LISTADO_LOCALIDADES = cargar_localidades()

class RemitoRequest(BaseModel):
    archivo_url: str = Field(..., description="URL de Google Drive del comprobante")


def obtener_id_drive(url: str) -> str:
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    match = re.search(r"id=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    return url


def normalizar_imagen(file_bytes: bytes, mime_type: str) -> tuple[bytes, str]:
    """
    Optimiza el consumo de tokens de entrada:
    1. Corrige orientación EXIF.
    2. Recorta bordes blancos e inútiles (Autotrim).
    3. Redimensiona a máximo 800px preservando legibilidad.
    4. Comprime a JPEG calidad 80.
    """
    if not mime_type.startswith("image/"):
        return file_bytes, mime_type
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img = ImageOps.exif_transpose(img)
        
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # Autotrim: Recorte inteligente de márgenes/fondos excesivamente claros
        bg = Image.new(img.mode, img.size, (255, 255, 255))
        diff = ImageChops.difference(img, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            img = img.crop(bbox)

        # Reducción óptima a 800px para minimizar grilla de píxeles/tokens
        img.thumbnail((800, 800))
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80, optimize=True)
        return buffer.getvalue(), "image/jpeg"
    except Exception:
        return file_bytes, mime_type


def descargar_archivo_drive(url: str) -> tuple[bytes, str]:
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
    prompt_instrucciones = f"""
    Sos un experto liquidador y auditor documental de Transporte Raosa S.R.L.

    INSTRUCCIÓN DE LECTURA Y ORIENTACIÓN:
    La imagen puede estar invertida o rotada. Determina la orientación correcta antes de extraer los datos.

    LISTADO OFICIAL DE LOCALIDADES Y CÓDIGOS:
    {LISTADO_LOCALIDADES}

    REGLAS DE EXTRACCIÓN Y FORMATO:
    - Remitente y Destinatario: Extrae razón social ('rs'), domicilio ('dom'), localidad ('loc'), provincia ('prov'), teléfono ('tel') y correo electrónico ('email') si figura. Si un dato no existe, coloca null de forma explícita.
    - Código de Localidad ('cod_loc'): Busca la localidad extraída en el Listado Oficial provisto arriba y asigna estrictamente el código numérico correspondiente (ej: "00001", "00051"). Si no se encuentra en la lista, coloca null.
    - CUIT / DNI: El campo 'cuit' debe extraerse SIN guiones ni separadores (11 dígitos para empresas, 8 para DNI). Si no hay CUIT o DNI explícito, coloca null.
    - Comprobante: Extrae el número de remito o factura ('nro') en un solo bloque continuo de 12 dígitos sin guiones (4 para punto de venta y 8 para número, ej: 000000003178).
    - Valor Declarado ('val_decl'): 
      * Si el comprobante es un remito, extrae el valor declarado que figure en el mismo.
      * Si el comprobante es una factura y los impuestos se encuentran discriminados, extrae **únicamente el subtotal sin impuestos (neto gravado)**, excluyendo el IVA y cualquier otro recargo impositivo. Si no existe, coloca null.
    - Flete:
      * 'cond_pago': 'ORIGEN' o 'DESTINO' únicamente si está explícitamente indicado en el comprobante. De lo contrario, null.
      * 'cr': true si indica contrarreembolso o C/R, de lo contrario false.
      * 'monto_cr': Monto numérico si hay contrarreembolso, de lo contrario null.
    - EXCLUYE COMPLETAMENTE: tipo de comprobante, fecha del comprobante, códigos postales, datos de carga, observaciones y total general de la factura.

    Estructura JSON ESTRICTA Y FIJA (Mantener siempre todas las claves devolviendo null si están vacías):
    {{
      "rem": {{
        "rs": str o null,
        "cuit": str o null,
        "dom": str o null,
        "loc": str o null,
        "cod_loc": str o null,
        "prov": str o null,
        "tel": str o null,
        "email": str o null
      }},
      "dest": {{
        "rs": str o null,
        "cuit": str o null,
        "dom": str o null,
        "loc": str o null,
        "cod_loc": str o null,
        "prov": str o null,
        "tel": str o null,
        "email": str o null
      }},
      "comprob": {{
        "nro": str o null,
        "val_decl": float o null
      }},
      "flete": {{
        "cond_pago": str o null,
        "cr": bool,
        "monto_cr": float o null
      }}
    }}

    Responde ÚNICAMENTE el JSON crudo respetando exactamente esta estructura completa, sin bloques markdown ni texto adicional.
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
    return {"status": "online", "service": "API Remitos y Facturas Raosa"}


@app.get("/procesar-remito")
@app.get("/procesar-remito/")
def procesar_remito_get(archivo_url: str = Query(..., description="URL de Drive del comprobante")):
    file_bytes, mime_type = descargar_archivo_drive(archivo_url)
    return procesar_con_gemini(file_bytes, mime_type)


@app.post("/procesar-remito")
@app.post("/procesar-remito/")
def procesar_remito_post(payload: RemitoRequest):
    file_bytes, mime_type = descargar_archivo_drive(payload.archivo_url)
    return procesar_con_gemini(file_bytes, mime_type)
