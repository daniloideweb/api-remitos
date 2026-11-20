import os
import re
import io
import json
import time
from datetime import datetime
from collections import defaultdict
import requests
from PIL import Image, ImageOps, ImageChops
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

app = FastAPI(
    title="API Extracción de Remitos y Facturas Raosa",
    description="Servicio de visión e IA con consolidación en Python de múltiples remitos, redespacho y persistencia en Sheets.",
    version="2.26.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client()

# URL del Webhook de Google Apps Script vinculado a la planilla de auditoría
WEBHOOK_SHEETS_URL = "https://script.google.com/macros/s/AKfycbwO2qETBxvbs9UQy98Lj3rqMZZgiqzp81sxxGAuLfI1yDKMMsQpFj0lcjevXYRW8p6OjA/exec"

# Registro en memoria para el conteo de llamados diarios (clave: "YYYY-MM-DD")
registro_llamados = defaultdict(int)


def registrar_llamado():
    hoy = datetime.now().strftime("%Y-%m-%d")
    registro_llamados[hoy] += 1


def enviar_a_google_sheets(resultado_ia: dict):
    try:
        comprob = resultado_ia.get("comprob", {})
        nro_principal = comprob.get("nro") or "N/D"
        otros = comprob.get("otros_nro") or []

        texto_comprobante = nro_principal
        if otros:
            texto_comprobante = f"{nro_principal} (+ {', '.join(otros)})"

        payload = {
            "comprobante": texto_comprobante,
            "remitente": resultado_ia.get("rem", {}).get("rs"),
            "destinatario": resultado_ia.get("dest", {}).get("rs"),
            "flete": resultado_ia.get("flete", {}).get("cond_pago")
        }
        requests.post(WEBHOOK_SHEETS_URL, json=payload, timeout=5)
    except Exception:
        pass


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
    if not mime_type.startswith("image/"):
        return file_bytes, mime_type
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img = ImageOps.exif_transpose(img)

        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        bg = Image.new(img.mode, img.size, (255, 255, 255))
        diff = ImageChops.difference(img, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            img = img.crop(bbox)

        img.thumbnail((1280, 1280))

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85, optimize=True)
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


def normalizar_nro_12_digitos(nro_raw) -> str | None:
    if not nro_raw:
        return None
    digitos = re.sub(r"\D", "", str(nro_raw))
    if not digitos:
        return None
    if len(digitos) == 12:
        return digitos
    if len(digitos) < 12:
        return digitos.zfill(12)
    # Si supera 12 dígitos, tomar los últimos 12 (4 PV + 8 número)
    return digitos[-12:]


def parsear_monto(valor) -> float | None:
    if valor is None or isinstance(valor, bool):
        return None
    if isinstance(valor, (int, float)):
        return float(valor)
    texto = str(valor).strip()
    if not texto:
        return None
    texto = re.sub(r"[^\d,.-]", "", texto)
    if not texto:
        return None
    if "," in texto and "." in texto:
        if texto.rfind(",") > texto.rfind("."):
            texto = texto.replace(".", "").replace(",", ".")
        else:
            texto = texto.replace(",", "")
    elif "," in texto:
        partes = texto.split(",")
        if len(partes[-1]) == 2:
            texto = texto.replace(".", "").replace(",", ".")
        else:
            texto = texto.replace(",", "")
    try:
        return float(texto)
    except ValueError:
        return None


def consolidar_comprobantes_en_python(resultado_ia: dict) -> dict:
    """
    Aplica las reglas de consolidación de forma determinista en Python:
    - Si hay remitos, descarta por completo las facturas.
    - Deduplica números de remito normalizados a 12 dígitos.
    - Asigna el primer remito a 'nro', los siguientes a 'otros_nro' y suma 'val_decl'.
    """
    detectados = resultado_ia.get("comprobantes_detectados") or []
    remitos_validos = []
    facturas_validas = []

    for item in detectados:
        if not isinstance(item, dict):
            continue
        tipo = str(item.get("tipo") or "").strip().upper()
        nro_norm = normalizar_nro_12_digitos(item.get("nro"))
        monto = parsear_monto(item.get("valor"))

        if not nro_norm:
            continue

        entrada = {"nro": nro_norm, "valor": monto}
        if "REM" in tipo:
            remitos_validos.append(entrada)
        elif "FAC" in tipo:
            facturas_validas.append(entrada)

    seleccionados = remitos_validos if remitos_validos else facturas_validas

    # Fallback por si el modelo no completó comprobantes_detectados
    if not seleccionados:
        comprob_previo = resultado_ia.get("comprob") or {}
        nro_fallback = normalizar_nro_12_digitos(comprob_previo.get("nro"))
        monto_fallback = parsear_monto(comprob_previo.get("val_decl"))
        if nro_fallback:
            seleccionados = [{"nro": nro_fallback, "valor": monto_fallback}]

    vistos = {}
    for item in seleccionados:
        nro_actual = item["nro"]
        valor_actual = item["valor"]
        if nro_actual not in vistos:
            vistos[nro_actual] = valor_actual
        else:
            if vistos[nro_actual] is None and valor_actual is not None:
                vistos[nro_actual] = valor_actual

    numeros_unicos = list(vistos.keys())
    nro_principal = numeros_unicos[0] if numeros_unicos else None
    otros_nro = numeros_unicos[1:] if len(numeros_unicos) > 1 else []

    valores_numericos = [v for v in vistos.values() if v is not None]
    total_val_decl = round(sum(valores_numericos), 2) if valores_numericos else None

    resultado_ia["comprob"] = {
        "nro": nro_principal,
        "otros_nro": otros_nro,
        "val_decl": total_val_decl
    }
    resultado_ia.pop("comprobantes_detectados", None)
    return resultado_ia


def procesar_con_gemini(file_bytes: bytes, mime_type: str) -> dict:
    prompt_instrucciones = f"""
    Sos un experto liquidador y auditor documental de Transporte Raosa S.R.L.

    INSTRUCCIÓN DE LECTURA Y ORIENTACIÓN:
    El archivo puede contener una o múltiples páginas/comprobantes, y algunas imágenes pueden estar invertidas o rotadas. Determina la orientación correcta antes de extraer los datos.

    LISTADO OFICIAL DE LOCALIDADES Y CÓDIGOS:
    {LISTADO_LOCALIDADES}

    REGLAS DE EXTRACCIÓN Y FORMATO:
    - Remitente y Destinatario:
      * Remitente ('rem'):
        Inspecciona el comprobante en busca de sellos o marcas de redespacho. Si detectas alguna de las siguientes empresas, debés asignar **estrictamente** los datos maestros oficiales de nuestra base (reemplazando al cliente original del membrete):
          1. **EXPRESO LUJAN DE CUYO S.A.** (Sello ovalado/circular rojo):
             - rs: "EXPRESO LUJAN DE CUYO S.A."
             - cuit: "30566258869"
             - dom: "CTC"
             - cod_loc: "00110"
          2. **ESA LOGISTICA S.A.** (Sello rectangular "DESPACHO POR CUENTA ESA LOGISTICA S.A."):
             - rs: "ESA LOGISTICA S.A."
             - cuit: "30707752250"
             - dom: "MONTEAGUDO 630"
             - cod_loc: "00110"
          3. **ENTREGAR S.A.** (Sello rectangular "CUENTA CORRIENTE ENTREGAR"):
             - rs: "ENTREGAR S.A."
             - cuit: "30707123873"
             - dom: "CORRIENTES Nº 531"
             - cod_loc: "00110"
        Si no existe ningún sello de redespacho, extrae los datos habituales del membrete superior del documento.
        Para el correo ('email') del remitente, extraelo si figura; si no, coloca null.
      * Teléfono ('tel'): Extrae un único número telefónico priorizando celular ('prefijo' sin el 0 inicial, 'numero' sin el prefijo móvil 15). Si no existe, null.
    - Código de Localidad ('cod_loc') para el Destinatario: Búscala en el Listado Oficial y asigna el código numérico correspondiente (ej: "00001"). Si no se encuentra, coloca null.
    - CUIT / DNI: El campo 'cuit' debe extraerse SIN guiones ni separadores (11 dígitos para empresas, 8 para DNI). Si no hay CUIT o DNI explícito, coloca null.

    - Comprobantes en TODAS las páginas ('comprobantes_detectados'):
      * Recorre **todas y cada una de las páginas** del archivo desde la primera hasta la última. No te detengas en la primera página.
      * En 'comprobantes_detectados', enumera en orden de aparición cada comprobante encontrado en cada página:
        - 'tipo': "REMITO" si es un remito, o "FACTURA" si es una factura.
        - 'nro': Número del comprobante en 12 dígitos continuos sin letras ni guiones (4 dígitos de punto de venta + 8 dígitos de número; ej. de "R00108-00217613" -> "010800217613").
        - 'valor': Si es REMITO, su valor declarado numérico (float). Si es FACTURA, su subtotal neto gravado sin IVA ni impuestos (float). Si no figura, null.
      * Si hay remitos y facturas en el legajo, igual listalos todos en 'comprobantes_detectados' con su 'tipo' correcto para que el sistema aplique la prioridad a remitos.

    - Flete:
      * 'cond_pago': **SOLO** si se detecta redespacho de **Expreso Luján de Cuyo S.A.**, **ESA Logística S.A.** o **Entregar S.A.**, asigna obligatoriamente `'ORIGEN'`. En los demás casos, extrae `'ORIGEN'` o `'DESTINO'` según figure en el comprobante, o usa `'DESTINO'` por defecto si no está indicado.
      * 'cr': true si indica contrarreembolso o C/R, de lo contrario false.
      * 'monto_cr': Monto numérico si hay contrarreembolso, de lo contrario null.
    - EXCLUYE COMPLETAMENTE: fecha del comprobante, códigos postales, datos de carga, observaciones y total general con impuestos de las facturas.

    Estructura JSON ESTRICTA Y FIJA (Mantener siempre todas las claves devolviendo null o [] según corresponda):
    {{
      "rem": {{
        "rs": str o null,
        "cuit": str o null,
        "dom": str o null,
        "cod_loc": str o null,
        "tel": {{
          "prefijo": str o null,
          "numero": str o null
        }},
        "email": str o null
      }},
      "dest": {{
        "rs": str o null,
        "cuit": str o null,
        "dom": str o null,
        "cod_loc": str o null,
        "tel": {{
          "prefijo": str o null,
          "numero": str o null
        }},
        "email": str o null
      }},
      "comprobantes_detectados": [
        {{
          "tipo": str,
          "nro": str o null,
          "valor": float o null
        }}
      ],
      "comprob": {{
        "nro": str o null,
        "otros_nro": [str],
        "val_decl": float o null
      }},
      "flete": {{
        "cond_pago": str,
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
            resultado_json = json.loads(response.text.strip())
            resultado_consolidado = consolidar_comprobantes_en_python(resultado_json)
            enviar_a_google_sheets(resultado_consolidado)
            return resultado_consolidado
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


@app.get("/estadisticas/diarias")
def estadisticas_diarias():
    hoy = datetime.now().strftime("%Y-%m-%d")
    return {
        "fecha_actual": hoy,
        "llamados_hoy": registro_llamados[hoy],
        "historico_por_dia": dict(registro_llamados)
    }


@app.get("/procesar-remito")
@app.get("/procesar-remito/")
def procesar_remito_get(archivo_url: str = Query(..., description="URL de Drive del comprobante")):
    registrar_llamado()
    file_bytes, mime_type = descargar_archivo_drive(archivo_url)
    return procesar_con_gemini(file_bytes, mime_type)


@app.post("/procesar-remito")
@app.post("/procesar-remito/")
def procesar_remito_post(payload: RemitoRequest):
    registrar_llamado()
    file_bytes, mime_type = descargar_archivo_drive(payload.archivo_url)
    return procesar_con_gemini(file_bytes, mime_type)
