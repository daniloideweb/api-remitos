import gc
import os
import re
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import defaultdict
import requests
from PIL import Image, ImageOps, ImageChops
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

try:
    import fitz  # PyMuPDF para separar/renderizar páginas de PDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

app = FastAPI(
    title="API Extracción de Remitos y Facturas Raosa",
    description="Servicio de visión e IA con renderizado PDF por lotes, consolidación en Python, redespacho y persistencia en Sheets.",
    version="2.30.0"
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
        requests.post(WEBHOOK_SHEETS_URL, json=payload, timeout=4)
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
    img = None
    bg = None
    diff = None
    buffer = None
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

        img.thumbnail((1024, 1024))

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80, optimize=True)
        return buffer.getvalue(), "image/jpeg"
    except Exception:
        return file_bytes, mime_type
    finally:
        for obj in (buffer, diff, bg, img):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        del buffer, diff, bg, img


def renderizar_rango_pdf(pdf_bytes: bytes, inicio: int, fin: int, dpi: int = 115) -> list[tuple[int, bytes, str]]:
    """
    Renderiza únicamente un rango acotado de páginas [inicio, fin) para mantener
    bajo el pico de RAM y permitir procesamiento concurrente por tanda.
    """
    lote = []
    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total = len(doc)
        limite = min(fin, total)
        zoom = dpi / 72.0
        matriz = fitz.Matrix(zoom, zoom)

        for idx in range(inicio, limite):
            pagina = doc[idx]
            pix = pagina.get_pixmap(matrix=matriz, alpha=False)
            img_bytes = pix.tobytes("jpeg")
            del pix
            norm_bytes, norm_mime = normalizar_imagen(img_bytes, "image/jpeg")
            del img_bytes
            lote.append((idx, norm_bytes, norm_mime))
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass
        del doc
    return lote


def _es_respuesta_valida(res: requests.Response) -> bool:
    if res.status_code != 200 or not res.content:
        return False
    cabecera = res.content[:150].lower()
    if res.content.startswith(b"<!DOCTYPE html>") or b"<html" in cabecera:
        return False
    return True


def descargar_archivo_drive(url: str) -> tuple[bytes, str]:
    file_id = obtener_id_drive(url)
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    # 1. Priorizar descarga directa del archivo original (conserva PDFs multipágina)
    download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&authuser=0"
    res = session.get(download_url, headers=headers, timeout=20)
    if "confirm=" in res.text:
        token = re.search(r"confirm=([0-9A-Za-z_]+)", res.text)
        if token:
            res = session.get(f"{download_url}&confirm={token.group(1)}", headers=headers, timeout=20)

    # 2. Respaldo vía lh3 si la descarga directa no devolvió archivo válido
    if not _es_respuesta_valida(res):
        alt_url = f"https://lh3.googleusercontent.com/d/{file_id}=s1600"
        res = session.get(alt_url, headers=headers, timeout=20)

    if not _es_respuesta_valida(res):
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

    if mime_type == "application/pdf":
        return res.content, mime_type

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


def _construir_config_rapida() -> types.GenerateContentConfig:
    try:
        return types.GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    except Exception:
        return types.GenerateContentConfig(
            response_mime_type="application/json"
        )


CONFIG_RAPIDA = _construir_config_rapida()


def llamar_gemini_una_parte(part_bytes: bytes, part_mime: str, es_pagina_adicional: bool = False) -> dict:
    contexto_hoja = (
        "Esta es una página adicional del mismo legajo. Enfocate en extraer los comprobantes presentes en esta hoja."
        if es_pagina_adicional
        else "Analiza este comprobante o primera hoja del envío y extrae los datos solicitados."
    )

    bloque_localidades = (
        f"LISTADO OFICIAL DE LOCALIDADES Y CÓDIGOS:\n{LISTADO_LOCALIDADES}"
        if not es_pagina_adicional
        else "Para 'cod_loc' en páginas adicionales, coloca null salvo que sea indispensable."
    )

    prompt_instrucciones = f"""
    Sos un experto liquidador y auditor documental de Transporte Raosa S.R.L.
    {contexto_hoja}

    INSTRUCCIÓN DE LECTURA Y ORIENTACIÓN:
    La imagen puede estar invertida o rotada. Determina la orientación correcta antes de extraer los datos.

    {bloque_localidades}

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

    - Comprobantes en esta hoja ('comprobantes_detectados'):
      * Enumera en orden de aparición cada comprobante visible en esta hoja:
        - 'tipo': "REMITO" si es un remito, o "FACTURA" si es una factura.
        - 'nro': Número del comprobante normalizado a 12 dígitos continuos sin letras ni guiones.
          REGLA SAP (MOLINOS RÍO DE LA PLATA + EXPRESO LUJÁN DE CUYO): Si el comprobante corresponde a Molinos Río de la Plata y está redespachado por Expreso Luján de Cuyo, **NO** uses el número superior del remito; tomá **estrictamente** el número del recuadro **"Entrega SAP"** (ej. si figura "0081039877", normalizalo a 12 dígitos -> "000081039877"). Si el comprobante tiene recuadro **"Entrega SAP"** en este escenario SAP, ese valor de Entrega SAP es el número obligatorio del comprobante. En los demás comprobantes, usá 4 dígitos de punto de venta + 8 dígitos de número (ej. de "R00108-00217613" -> "010800217613").
        - 'valor': Si es REMITO, su valor declarado numérico (float) si figura explícitamente; en remitos de Molinos redespachados por Expreso Luján de Cuyo donde no haya un valor declarado explícito o solo figure el total de bultos/unidades, devolvé null salvo que exista un importe monetario expreso. Si es FACTURA, su subtotal neto gravado sin IVA ni impuestos (float). Si no figura, null.

    - Flete:
      * 'cond_pago': **SOLO** si se detecta redespacho de **Expreso Luján de Cuyo S.A.**, **ESA Logística S.A.** o **Entregar S.A.**, asigna obligatoriamente `'ORIGEN'`. En los demás casos, extrae `'ORIGEN'` o `'DESTINO'` según figure en el comprobante, o usa `'DESTINO'` por defecto si no está indicado.
      * 'cr': true si indica contrarreembolso o C/R, de lo contrario false.
      * 'monto_cr': Monto numérico si hay contrarreembolso, de lo contrario null.
    - EXCLUYE COMPLETAMENTE: fecha del comprobante, códigos postales, datos de carga, observaciones y total general con impuestos de las facturas.

    Estructura JSON ESTRICTA Y FIJA (Mantener siempre todas las claves devolviendo null si están vacías):
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
      "flete": {{
        "cond_pago": str,
        "cr": bool,
        "monto_cr": float o null
      }}
    }}

    Responde ÚNICAMENTE el JSON crudo respetando exactamente esta estructura completa, sin bloques markdown ni texto adicional.
    """

    part_archivo = types.Part.from_bytes(data=part_bytes, mime_type=part_mime)
    part_texto = types.Part.from_text(text=prompt_instrucciones)
    contenido = types.Content(role="user", parts=[part_archivo, part_texto])

    max_intentos = 2
    ultimo_error = None

    for _ in range(max_intentos):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=[contenido],
                config=CONFIG_RAPIDA
            )
            return json.loads(response.text.strip())
        except Exception as e:
            ultimo_error = str(e)

            if any(err in ultimo_error for err in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"]):
                espera = 1.5
                match_delay = re.search(r"retryDelay':\s*'(\d+)", ultimo_error)
                if match_delay:
                    espera = min(float(match_delay.group(1)) + 0.3, 4.0)
                else:
                    match_sec = re.search(r"retry in (\d+(?:\.\d+)?)s", ultimo_error)
                    if match_sec:
                        espera = min(float(match_sec.group(1)) + 0.3, 4.0)

                time.sleep(espera)
                continue

            break

    raise HTTPException(
        status_code=500,
        detail=f"Error en el procesamiento de visión/IA: {ultimo_error}"
    )


def consolidar_resultados_paginas(resultados_por_pagina: list[dict]) -> dict:
    """
    Consolida los datos extraídos de una o múltiples páginas en Python:
    - Toma el bloque 'rem', 'dest' y 'flete' más completo disponible.
    - Si hay remitos en cualquier página, descarta por completo las facturas.
    - Deduplica números de remito normalizados a 12 dígitos.
    - Asigna el primer remito a 'nro', los demás a 'otros_nro' y suma 'val_decl'.
    """
    print(f"[CONSOLIDAR] Cantidad de hojas recibidas: {len(resultados_por_pagina)}", flush=True)
    base_rem = None
    base_dest = None
    base_flete = None

    todos_detectados = []

    for idx, res in enumerate(resultados_por_pagina):
        if not isinstance(res, dict):
            print(f"[CONSOLIDAR] Hoja {idx + 1} sin dict válido: {res}", flush=True)
            continue

        detectados_hoja = res.get("comprobantes_detectados") or []
        print(f"[CONSOLIDAR] Hoja {idx + 1} detectados: {detectados_hoja}", flush=True)

        rem_actual = res.get("rem")
        if isinstance(rem_actual, dict) and rem_actual.get("rs"):
            if base_rem is None or rem_actual.get("rs") in {
                "EXPRESO LUJAN DE CUYO S.A.",
                "ESA LOGISTICA S.A.",
                "ENTREGAR S.A."
            }:
                base_rem = rem_actual

        dest_actual = res.get("dest")
        if isinstance(dest_actual, dict) and dest_actual.get("rs"):
            if base_dest is None:
                base_dest = dest_actual

        flete_actual = res.get("flete")
        if isinstance(flete_actual, dict):
            if base_flete is None or flete_actual.get("cond_pago") == "ORIGEN":
                base_flete = flete_actual

        for item in detectados_hoja:
            if isinstance(item, dict):
                todos_detectados.append(item)

    remitos_validos = []
    facturas_validas = []

    for item in todos_detectados:
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

    print(f"[CONSOLIDAR] Final -> nro={nro_principal}, otros={otros_nro}, val_decl={total_val_decl}", flush=True)

    return {
        "rem": base_rem or {
            "rs": None,
            "cuit": None,
            "dom": None,
            "cod_loc": None,
            "tel": {"prefijo": None, "numero": None},
            "email": None
        },
        "dest": base_dest or {
            "rs": None,
            "cuit": None,
            "dom": None,
            "cod_loc": None,
            "tel": {"prefijo": None, "numero": None},
            "email": None
        },
        "comprob": {
            "nro": nro_principal,
            "otros_nro": otros_nro,
            "val_decl": total_val_decl
        },
        "flete": base_flete or {
            "cond_pago": "DESTINO",
            "cr": False,
            "monto_cr": None
        }
    }


def procesar_con_gemini(file_bytes: bytes, mime_type: str, background_tasks: BackgroundTasks | None = None) -> dict:
    t_inicio = time.perf_counter()
    print(f"[PROCESAR] MIME recibido: {mime_type}, HAS_FITZ={HAS_FITZ}", flush=True)
    resultados_por_idx = {}

    if mime_type == "application/pdf" and HAS_FITZ:
        try:
            doc_tmp = fitz.open(stream=file_bytes, filetype="pdf")
            total_hojas = len(doc_tmp)
            doc_tmp.close()
            del doc_tmp

            tamano_lote = 4
            with ThreadPoolExecutor(max_workers=tamano_lote) as executor:
                for inicio in range(0, total_hojas, tamano_lote):
                    fin = min(inicio + tamano_lote, total_hojas)
                    lote = renderizar_rango_pdf(file_bytes, inicio, fin)
                    print(f"[PROCESAR] Tanda hojas {inicio + 1}-{fin}/{total_hojas}...", flush=True)

                    futuros = {
                        executor.submit(
                            llamar_gemini_una_parte,
                            hoja_bytes,
                            hoja_mime,
                            (idx > 0)
                        ): idx
                        for idx, hoja_bytes, hoja_mime in lote
                    }

                    for futuro in as_completed(futuros):
                        idx_resuelto = futuros[futuro]
                        resultados_por_idx[idx_resuelto] = futuro.result()

                    del lote, futuros

            resultados_hojas = [resultados_por_idx[i] for i in sorted(resultados_por_idx.keys())]
        except Exception as exc:
            print(f"[PROCESAR] Fallback PDF directo tras error en tandas: {exc}", flush=True)
            resultados_hojas = [llamar_gemini_una_parte(file_bytes, mime_type, es_pagina_adicional=False)]
    else:
        print("[PROCESAR] Procesando hoja 1/1...", flush=True)
        resultados_hojas = [
            llamar_gemini_una_parte(file_bytes, mime_type, es_pagina_adicional=False)
        ]

    del file_bytes, resultados_por_idx
    gc.collect()

    resultado_final = consolidar_resultados_paginas(resultados_hojas)
    if background_tasks is not None:
        background_tasks.add_task(enviar_a_google_sheets, resultado_final)
    else:
        enviar_a_google_sheets(resultado_final)

    print(f"[PROCESAR] Tiempo total de procesamiento: {time.perf_counter() - t_inicio:.2f}s", flush=True)
    return resultado_final


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
def procesar_remito_get(
    background_tasks: BackgroundTasks,
    archivo_url: str = Query(..., description="URL de Drive del comprobante")
):
    registrar_llamado()
    file_bytes, mime_type = descargar_archivo_drive(archivo_url)
    return procesar_con_gemini(file_bytes, mime_type, background_tasks)


@app.post("/procesar-remito")
@app.post("/procesar-remito/")
def procesar_remito_post(payload: RemitoRequest, background_tasks: BackgroundTasks):
    registrar_llamado()
    file_bytes, mime_type = descargar_archivo_drive(payload.archivo_url)
    return procesar_con_gemini(file_bytes, mime_type, background_tasks)
