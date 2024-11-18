import pandas as pd
from joblib import dump
from dotenv import load_dotenv
import os

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(df, target_column):
    # Separar variables predictoras y objetivo
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identificar columnas numéricas y categóricas
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # Crear transformadores para cada tipo de variable
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Aplicar preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor, X, y



def preprocess_and_train(dataset_path, target_column, model_name, trials):
    print(f"Dataset Path: {dataset_path}")
    print(f"Target Column: {target_column}")
    print(f"Model Name: {model_name}")
    print(f"Trials: {trials}")

    # Cargar datos
    print(f"Intentando cargar dataset desde: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    print("Dataset cargado exitosamente:")
    print(df.head())

    # Preprocesamiento
    preprocessor, X, y = preprocess_data(df, target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Selección del modelo
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB()
    }

    if model_name not in models:
        raise ValueError(f"Modelo '{model_name}' no está soportado. Elija uno de {list(models.keys())}.")

    base_model = models[model_name]

    # Crear pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', base_model)])

    # Optimización de hiperparámetros
    print(f"Optimización de hiperparámetros para {model_name}...")
    optimized_pipeline = optimize_hyperparameters(pipeline, model_name, X_train, y_train, trials)

    # Entrenar modelo optimizado
    print(f"Entrenando modelo optimizado {model_name}...")
    optimized_pipeline.fit(X_train, y_train)

    # Evaluación
    y_pred = optimized_pipeline.predict(X_test)
    print("Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    # Guardar modelo
    dump(optimized_pipeline, "model.pkl")

    # Verificar si el archivo fue creado y mostrar su ruta
    model_path = os.path.abspath("model.pkl")
    if os.path.exists(model_path):
        print(f"El modelo fue guardado correctamente en: {model_path}")
    else:
        raise FileNotFoundError(f"No se generó el archivo model.pkl en {model_path}.")





def optimize_hyperparameters(model, model_name, X_train, y_train, trials):
    # Diccionario de hiperparámetros
    param_distributions = {
        "RandomForest": {"model__n_estimators": [10, 50, 100]},
        "GradientBoosting": {"model__learning_rate": [0.01, 0.1, 1.0], "model__n_estimators": [50, 100, 150]},
        "SVM": {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf"]},
        "KNN": {"model__n_neighbors": [3, 5, 7, 10]},
        "NaiveBayes": {} 
    }

    search = RandomizedSearchCV(
        model,
        param_distributions.get(model_name, {}),
        n_iter=trials,
        cv=3,
        random_state=42,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_

if __name__ == "__main__":
    # Cargar variables desde el archivo .env
    load_dotenv()

    # Leer parámetros desde el .env
    dataset_path = os.getenv("DATASET")
    target_column = os.getenv("TARGET").strip('"')
    model_name = os.getenv("MODEL")
    trials = int(os.getenv("TRIALS"))

    print(f"Dataset Path: {dataset_path}")
    print(f"Target Column: {target_column}")
    print(f"Model Name: {model_name}")
    print(f"Trials: {trials}")

    # Llamar a la función principal
    preprocess_and_train(dataset_path, target_column, model_name, trials)
