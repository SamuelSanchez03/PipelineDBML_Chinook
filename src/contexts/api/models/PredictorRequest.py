from pydantic import BaseModel, field_validator

class PredictorRequest(BaseModel):
    dominio_correo: str
    pais_origen: str
    ciudad_origen:str
    
    @field_validator("dominio_correo", "pais_origen", "ciudad_origen")
    @classmethod
    def no_vacio(cls, valor: str) -> str:
        if not valor or not valor.strip():
            raise ValueError("El campo no puede estar vacío")
        return valor.strip()