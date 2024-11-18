import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump
from dotenv import load_dotenv
import os

def preprocess_and_train(dataset_path, target_column, model_name, trials):
    # Cargar datos
    df = pd.read_parquet(dataset_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar modelo (por simplicidad, solo RandomForest por ahroa)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluar modelo
    y_pred = model.predict(X_test)
    print("Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    # Guardar modelo
    dump(model, "model.pkl")
    print("Modelo guardado como 'model.pkl'.")

if __name__ == "__main__":

    # Cargar variables desde el archivo .env
    load_dotenv()
    dataset_path = os.getenv("DATASET")
    target_column = os.getenv("TARGET").strip('"')
    model_name = os.getenv("MODEL")
    trials = int(os.getenv("TRIALS"))

    print(f"Columna objetivo: {target_column}")
    preprocess_and_train(dataset_path, target_column, model_name, trials)
