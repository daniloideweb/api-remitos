import time

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
    - Responder ÚNICAMENTE el JSON crudo, sin bloques markdown.
    """

    part_archivo = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
    part_texto = types.Part.from_text(text=prompt_instrucciones)
    contenido = types.Content(role="user", parts=[part_archivo, part_texto])
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.1
    )

    # Lista de modelos con fallback en cascada ante alta demanda
    modelos = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-2.5-pro"]
    ultimo_error = None

    for modelo in modelos:
        for intento in range(2):  # Hasta 2 intentos por modelo
            try:
                response = client.models.generate_content(
                    model=modelo,
                    contents=[contenido],
                    config=config
                )
                return json.loads(response.text.strip())
            except Exception as e:
                error_msg = str(e)
                ultimo_error = error_msg
                # Si es saturación (503), aguarda 2 segundos antes de reintentar
                if "503" in error_msg or "UNAVAILABLE" in error_msg:
                    time.sleep(2)
                    continue
                else:
                    break  # Si es otro tipo de error, pasa al siguiente modelo

    raise HTTPException(
        status_code=500,
        detail=f"Servidores de IA ocupados tras varios intentos: {ultimo_error}"
    )
