import os
import joblib
from fastapi import HTTPException
import pandas as pd

from src.contexts.api.models import PredictorRequest

class TrainModelController:
    def execute(self, request: PredictorRequest):
        print(request)
        dominio_correo = request.dominio_correo
        pais_origen = request.pais_origen
        ciudad_origen = request.ciudad_origen
       
        ruta_modelo = os.getenv("MODELO_ENTRENADO")

        if not os.path.exists(ruta_modelo):
            raise HTTPException(
                status_code=503,
                detail="El modelo aún no ha sido entrenado. Espera a que el cron finalice el primer entrenamiento."
            )

        modelo = joblib.load(ruta_modelo)

        
        df_input = pd.DataFrame([{
            "dominio_correo": request.dominio_correo,
            "pais_origen": request.pais_origen,
            "ciudad_origen": request.ciudad_origen
        }])
        
        resultado = modelo.predict(df_input)
        
        print(f"Predicción para X={df_input}: {resultado[0]}")
        
        return {"status": "OK", "result": resultado[0]}