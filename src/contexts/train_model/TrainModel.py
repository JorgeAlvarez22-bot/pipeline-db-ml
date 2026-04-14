import numpy as np
import joblib
import psycopg2
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

class TrainModel:

    def entrenarModelo():
        load_dotenv("/app/.env")
        USER = os.getenv("SUPABASE_USER")
        PASSWORD = os.getenv("SUPABASE_PASSWORD")
        HOST = os.getenv("SUPABASE_HOST")
        PORT = os.getenv("SUPABASE_PORT")
        DBNAME = os.getenv("SUPABASE_DBNAME")

        if PORT is None:
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
                    cursor.execute('SELECT email, country, city, genre FROM "train_model";')
                    rows = cursor.fetchall()
                    print(f"Filas recuperadas: {len(rows)}")

        except Exception as e:
            print(f"Error al conectar o recuperar datos: {e}")
            return

        if not rows:
            print("No se recuperaron filas. Abortando.")
            return

        print(rows[:2])

        data_array = np.array(rows)
        X = data_array[:, :3]
        y = data_array[:, 3]

        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_encoded = encoder.fit_transform(X)

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        x_train, x_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)

        joblib.dump(model, str(os.getenv("MODELO_ENTRENADO")))
        joblib.dump(encoder, str(os.getenv("ENCODER_ENTRENADO")))
        joblib.dump(label_encoder, str(os.getenv("LABEL_ENCODER_ENTRENADO")))

        print("modelo entrenado y guardado correctamente")