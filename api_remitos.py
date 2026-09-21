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

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.1
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
                # Espera más espaciada (4s, 7s, 10s) para sortear el pico de Google
                time.sleep(4 + (intento * 3))
                continue
            break

    raise HTTPException(
        status_code=500,
        detail=f"Error en el procesamiento de visión/IA: {ultimo_error}"
    )
