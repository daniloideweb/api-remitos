def procesar_con_gemini(file_bytes: bytes, mime_type: str) -> dict:
    prompt_instrucciones = """
    Sos un especialista senior en liquidación y control documental de transporte de cargas y encomiendas.
    
    ATENCIÓN A LA ORIENTACIÓN:
    La imagen puede estar rotada, inclinada o invertida (de cabeza). Leé el texto orientándolo mentalmente de manera correcta antes de extraer los datos.

    CONTEXTO DEL COMPROBANTE:
    Este documento es una guía o remito de transporte de encomiendas que contiene dos fuentes de información:
    1. Un comprobante base (impreso o manuscrito con campos como Remitente, Destinatario, Domicilio, Localidad, Descripción, Totales).
    2. Opcionalmente, una etiqueta térmica blanca pegada (etiqueta de bulto/despacho con código de barras, bultos 'Bul:', remitente/destinatario y agencia).

    REGLAS DE EXTRACCIÓN:
    - Si hay etiqueta adhesiva, contrastá los datos manuscritos con los impresos en la etiqueta para mayor exactitud ortográfica (ej. nombres, número de remito/guía, bultos y destino).
    - Identificá claramente quién envía (REMITENTE) y quién recibe (DESTINATARIO). No los inviertas.
    - Números manuscritos: Prestá especial atención a CUIT/DNI, teléfonos, códigos postales y montos totales o valores declarados. Si un importe está tachado y reescrito, tomá el valor final legible.
    - Condición de pago: Si figura 'ORIGEN' o pagado en origen -> 'ORIGEN'. Si es 'DESTINO' o flete a cobrar -> 'DESTINO'.
    - Contrarreembolso: Indicá true sólo si explícitamente figura C/R o contrarreembolso con su importe monetario a cobrar.

    Generá EXCLUSIVAMENTE un objeto JSON válido con esta estructura estricta:
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

    Respondé únicamente el JSON crudo, sin bloques markdown ```json.
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
