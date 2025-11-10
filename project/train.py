import warnings
import argparse
import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from openai import OpenAI
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
model_name = "gemini-2.5-flash"

# Inicializa el cliente de OpenAI (asegúrate de que OPENAI_API_KEY esté en tus variables de entorno)
gemini_model = OpenAI(
    api_key=gemini_api_key, 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )


if not gemini_api_key:
    raise ValueError("La variable de entorno 'GEMINI_API_KEY' no está configurada.")

# ---------- Funciones de evaluación de métricas ----------
def eval_metrics(actual, pred):
    """Calcula métricas clave para regresión."""
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# ---------- Funciones de generación de AI ----------
def generate_explanation(model_metrics, feature_importances):
    """Genera una explicación de la performance del modelo usando Gen AI."""

    # Crea el prompt para la IA
    user_prompt = f"""
    Eres un analista de datos. Genera un insight conciso (máximo 50 palabras) sobre el rendimiento del modelo 
    de regresión de calidad de vino, utilizando las siguientes métricas y las características más importantes.

    Métricas del Modelo:
    - RMSE (Error Cuadrático Medio Raíz): {model_metrics['rmse']:.4f}
    - R2 (Coeficiente de Determinación): {model_metrics['r2']:.4f}

    Importancia de las Características (Top 3):
    1. {feature_importances[0][0]}: {feature_importances[0][1]:.2f}
    2. {feature_importances[1][0]}: {feature_importances[1][1]:.2f}
    3. {feature_importances[2][0]}: {feature_importances[2][1]:.2f}

    Conclusión:
    """

    try:
        response = gemini_model.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Eres un experto en Machine Learning y explicas los modelos de forma clara."},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error al generar explicación con Gen AI: {e}")
        return "Explicación Gen AI no disponible debido a un error de conexión o API Key."

# --- 1. Carga del dataset ---
df = pd.read_csv("../data/winequality-white.csv", sep=";")

# Dividir datos
X = df.drop('quality', axis=1)
y = df['quality']

# Split de datos entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Tamaño de entrenamiento: {X_train.shape}, Tamaño de prueba: {X_test.shape}")


print(X_train.head())

print("---"*30)

print(y_train.head())

if __name__ == "__main__":
    #warnings.filterwarnings("ignore")
    np.random.seed(42)

    # --- 2. Configuración de Argumentos y Experimento MLflow ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    # --- 4. Entrenamiento del Modelo ---
    rf = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
    rf.fit(X_train, y_train)

    # --- 5. Predicciones ---
    y_pred = rf.predict(X_test)

    # --- 6. Evaluación de Métricas ---
    rmse, mae, r2 = eval_metrics(y_test, y_pred)
    print(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}")

    # mlflow.log_param("n_estimators", args.n_estimators)
    # mlflow.log_param("max_depth", args.max_depth)
    # mlflow.log_param("random_state", args.random_state)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # --- 7. Importancia de Características ---
    feature_importances = sorted(
        zip(X.columns, rf.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )

    # --- 8. Generación de Explicación con Gen AI ---
    run_id = mlflow.active_run().info.run_id
    model_metrics = {'rmse': rmse, 'r2': r2}
    explanation = generate_explanation(model_metrics, feature_importances)
    print("Explicación Generada por Gen AI:")
    print(explanation)
    # --- 9. Registro en MLflow ---
    mlflow.log_params(vars(args))
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(rf, "random_forest_model", input_example=X_train.iloc[:5])
    version = mlflow.register_model(model_uri=f"runs:/{run_id}/random_forest_model", name="random_forest_model")

    client = MlflowClient()
    client.set_model_version_tag(
        name="random_forest_model",
        version=version.version,
        key="status",
        value="stagging"
    )


    # Registrar la explicación generada
    mlflow.log_text(explanation, "model_explanation.txt")