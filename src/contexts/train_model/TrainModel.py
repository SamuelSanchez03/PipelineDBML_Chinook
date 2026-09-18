import numpy as np
import joblib
import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

class TrainModel:

    def entrenarModelo():

        #se usaron las credeciales para ingresar de manera ocacional (Transaction pooler)
        load_dotenv("/app/.env")
        USER = os.getenv("SUPABASE_USER")
        PASSWORD = os.getenv("SUPABASE_PASSWORD")
        HOST = os.getenv("SUPABASE_HOST")
        PORT = os.getenv("SUPABASE_PORT")
        DBNAME = os.getenv("SUPABASE_DBNAME")
        

        if(PORT== None):
            print("no se lee el env")
            return
        else:
            print("si se lee en env")


        try:
            with psycopg2.connect(
                user=USER,
                password=PASSWORD,
                host=HOST,
                port=PORT,
                dbname=DBNAME
            ) as connection:
                with connection.cursor() as cursor:
                    # Consulta SQL
                    cursor.execute('SELECT dominio_correo, pais_origen, ciudad_origen, genero_musical FROM usuario;')
                    rows = cursor.fetchall()  # devuelve una lista de tuplas [(x1,y1),(x2,y2),...]
                    
                    print(f"Filas recuperadas: {len(rows)}")

        except Exception as e:
            print(f"Error al conectar o recuperar datos: {e}")
            return
        
        if not rows:
            print("No se recuperaron filas de la base de datos. Abortando entrenamiento.")
            return
        else:
            print(rows[:2])
            

        columnas = ["dominio_correo", "pais_origen", "ciudad_origen", "genero_musical"]
        df = pd.DataFrame(rows, columns=columnas)

        # Features (X) y target (y)
        X = df[["dominio_correo", "pais_origen", "ciudad_origen"]]
        y = df["genero_musical"]
        
        columnas_categoricas = ["dominio_correo", "pais_origen", "ciudad_origen"]

        preprocesador = ColumnTransformer(
            transformers=[
                ("onehot", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas)
            ]
        )

        pipeline = Pipeline(steps=[
            ("preprocesador", preprocesador),
            ("clasificador", RandomForestClassifier(random_state=42))
        ])
        
        
        #entrenar el modelo
        pipeline.fit(X, y)
        
        accuracy = pipeline.score(X, y)
        print(f"accuracy (train): {accuracy:.4f}")
    
        joblib.dump(pipeline, str(os.getenv("MODELO_ENTRENADO")))
        print("modelo entrenado")
        
