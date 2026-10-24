import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional
import google.generativeai as genai
from PIL import Image
import io

# Inicializar la API de Gemini con la clave del entorno
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("No se encontró la variable de entorno GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="API Remitos Raosa", version="2.0")

# Esquema Pydantic optimizado para minimizar tokens de salida
class Remitente(BaseModel):
    rs: Optional[str] = Field(None, description="Razón social del remitente")
    cuit: Optional[str] = Field(None, description="CUIT del remitente")
    dom: Optional[str] = Field(None, description="Domicilio del remitente")
    loc: Optional[str] = Field(None, description="Localidad del remitente")
    prov: Optional[str] = Field(None, description="Provincia del remitente")
    tel: Optional[str] = Field(None, description="Teléfono del remitente")
    email: Optional[str] = Field(None, description="Correo electrónico del remitente")

class Destinatario(BaseModel):
    rs: Optional[str] = Field(None, description="Razón social del destinatario")
    cuit: Optional[str] = Field(None, description="CUIT del destinatario")
    dom: Optional[str] = Field(None, description="Domicilio del destinatario")
    loc: Optional[str] = Field(None, description="Localidad del destinatario")
    prov: Optional[str] = Field(None, description="Provincia del destinatario")
    tel: Optional[str] = Field(None, description="Teléfono del destinatario")
    email: Optional[str] = Field(None, description="Correo electrónico del destinatario")

class Comprobante(BaseModel):
    tipo: Optional[str] = Field("REMITO", description="Tipo de comprobante")
    nro: Optional[str] = Field(None, description="Número de remito en formato continuo de 12 dígitos sin guiones: 4 de punto de venta y 8 de número (ej: 000100047496)")
    val_decl: Optional[float] = Field(None, description="Valor declarado")

class Flete(BaseModel):
    cond_pago: Optional[str] = Field(None, description="Condición de pago")
    cr: Optional[bool] = Field(False, description="Contrariareembolso True o False")
    monto_cr: Optional[float] = Field(None, description="Monto de contrarreembolso")

class RemitoResponse(BaseModel):
    rem: Remitente
    dest: Destinatario
    comprob: Comprobante
    flete: Flete

@app.post("/procesar-remito", response_model=RemitoResponse)
async def procesar_remito(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Usar el modelo gemini-3.6-flash para extracción veloz y económica
        model = genai.GenerativeModel('gemini-3.6-flash')
        
        prompt = (
            "Extrae los datos de este remito respetando estrictamente la estructura solicitada. "
            "El número de remito ('nro') debe unirse en un solo bloque de 12 dígitos sin guiones "
            "(4 dígitos para punto de venta y 8 para el número, rellenando con ceros a la izquierda si es necesario, ej: 000100047496). "
            "No incluyas fecha, códigos postales, datos de carga, observaciones ni totales de flete."
        )
        
        response = model.generate_content(
            [image, prompt],
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=RemitoResponse,
                temperature=0.1
            )
        )
        
        import json
        data = json.loads(response.text)
        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "API Remitos Raosa activa y operando"}
