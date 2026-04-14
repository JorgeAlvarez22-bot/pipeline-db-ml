import os
import joblib
import numpy as np
import os

from src.contexts.api.models import PredictorRequest



class TrainModelController:
    def execute(self, request: PredictorRequest):
        print(request)
        email=request.email
        country = request.country
        city= request.city
       
        # Cargar el modelo desde el archivo
        modelo_cargado = joblib.load(os.getenv("MODELO_ENTRENADO"))
        encoder= joblib.load(os.getenv("ENCODER_ENTRENADO"))
        label_encoder = joblib.load(os.getenv("LABEL_ENCODER_ENTRENADO"))


        # Creando y codificando el dato
        nuevo_dato = encoder.transform(np.array([[email, country, city]]))

        # Mandando el dato para hacer la predicción
        result = modelo_cargado.predict(nuevo_dato)
        prediction= label_encoder.inverse_transform(result)
        print(f"Predicción: {prediction[0]}")
        
        return {"status": "OK", "result": prediction[0]}
        