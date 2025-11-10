import gradio as gr
import pandas as pd
import numpy as np
import mlflow.pyfunc
import os
import io
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv

# Carga variables de entorno
load_dotenv()

# --- Configuración Global y MLflow ---

model_name = "gemini-2.5-flash"
MODEL_NAME = "random_forest_model"
TAG_KEY = "status"
TAG_VALUE = "production"
MODEL_NAME = "random_forest_model"

# *** CONFIGURACIÓN CLAVE para conectar con tu servidor MLflow ***
# Esto resuelve el 'FutureWarning' y el error de modelo no encontrado.
MLFLOW_SERVER_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_SERVER_URI)
# *************************************************************

FEATURE_COLUMNS = [
    "fixed acidity", "volatile acidity", "citric acid", 
    "residual sugar", "chlorides", "free sulfur dioxide", 
    "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
]
INITIAL_VALUES = {
    "fixed acidity": 7.0, "volatile acidity": 0.27, "citric acid": 0.36, 
    "residual sugar": 20.7, "chlorides": 0.045, "free sulfur dioxide": 45.0, 
    "total sulfur dioxide": 170.0, "density": 1.001, "pH": 3.0, 
    "sulphates": 0.45, "alcohol": 8.8
}


# --- Carga del Modelo ---
try:
    # Se conecta al servidor usando la URI que acabamos de configurar
    client = mlflow.tracking.MlflowClient()

    filter = f"name='{MODEL_NAME}' AND tags.{TAG_KEY}='{TAG_VALUE}'"

    results = client.search_model_versions(filter_string=filter, order_by=["creation_timestamp DESC"])

    if not results:
        raise Exception(f"No se encontró ninguna versión con el tag {TAG_KEY}='{TAG_VALUE}'.")
    
    latest_version = results[0]
    logged_model_uri = f'models:/{latest_version.name}/{latest_version.version}'
    loaded_model = mlflow.pyfunc.load_model(logged_model_uri)

    MODEL_STATUS = f"Modelo Versión ({latest_version.version}, Tag: {TAG_VALUE}) - Conectado a {MLFLOW_SERVER_URI}"
except Exception as e:
    print(f"Error al cargar modelo de MLflow Registry: {e}. Usando modelo dummy.")
    MODEL_STATUS = "Error al cargar modelo. Usando Dummy."
    class DummyModel:
        def predict(self, df):
            return np.array([6.0] * len(df))
    model = DummyModel()


# --- Funciones Lógicas de Gradio ---

def get_gemini_client():
    """Inicializa y devuelve el cliente de OpenAI/Gemini."""
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    model = genai.GenerativeModel(model_name)

    return model

# ... (Función get_gen_ai_explanation se mantiene igual)
def get_gen_ai_explanation(input_data, prediction):
    """Genera una explicación Gen AI para la predicción individual."""

    gemini_model = get_gemini_client()
    characteristics = ", ".join([
        f"{col}: {val}" for col, val in zip(FEATURE_COLUMNS, input_data.iloc[0])
    ])
    
    print(f"Generating Gen AI explanation for prediction: {prediction[0]:.2f} with characteristics: {characteristics}")

    prompt = f"""
    Un modelo predijo que la calidad de un vino es de {prediction[0]:.2f}.
    Propiedades: {characteristics}. Genera una explicación de una sola frase (máx. 25 palabras) que justifique la calidad.
    """

    try:

        response = gemini_model.generate_content(prompt)

        print(f"Gen AI raw response: {response}\n")

        print(f"Gen AI response: {response.text.strip()}")

        return response.text.strip()
    except Exception as e:
        return f"Error Gen AI: Verifique la API Key de OpenAI. {e}"


def predict_wine_quality(*args):
    """Función de predicción para un único punto de datos (Single Input) con MLflow Tracking."""
    
    # Inicia un nuevo RUN de MLflow para esta predicción
    # Usa un tag para identificar que esta predicción viene de la App Gradio
    with mlflow.start_run(run_name="Single_Prediction", tags={"source": "Gradio_App", "type": "single"}):
        
        input_values = list(args)

        float_values = [float(v) for v in input_values]

        input_df = pd.DataFrame([float_values], columns=FEATURE_COLUMNS)

        prediction = loaded_model.predict(input_df)

        quality_score = prediction[0]

        # 1. Registrar Parámetros (Inputs)
        input_params = {k: v for k, v in zip(FEATURE_COLUMNS, float_values)}
        mlflow.log_params(input_params)
        
        # 2. Registrar Métrica (Output)
        mlflow.log_metric("predicted_quality", quality_score)
        
        # 3. Generar Explicación Gen AI
        explanation = get_gen_ai_explanation(input_df, prediction)
        mlflow.log_text(explanation, "gen_ai_explanation.txt") # Registrar la explicación como artefacto
        
        # 4. Formatear y Determinar Insight
        formatted_prediction = f"Predicción de Calidad: {quality_score:.2f} / 10"

        if quality_score >= 7.0:
            insight = "¡Excelente! Un vino de alta calidad."
        elif quality_score >= 5.0:
            insight = "Calidad promedio. El modelo sugiere un vino bebible."
        else:
            insight = "Baja calidad. Se recomienda precaución."
        
        return formatted_prediction, insight, explanation


def predict_from_csv(csv_file):
    """Función de predicción para datos en lote (CSV Upload) con MLflow Tracking."""
    summary_default = "Inicie la predicción cargando un archivo CSV."
    if csv_file is None:
        return "Error: No se ha cargado ningún archivo.", pd.DataFrame()

    with mlflow.start_run(run_name="Batch_Prediction", tags={"source": "Gradio_App", "type": "batch"}) as run:
        try:
            df_input = pd.read_csv(csv_file.name, sep=";")
            print(f"CSV cargado: {df_input}.")
        except Exception as e:
            return f"Error al leer el CSV: {e}", pd.DataFrame()
        
        print(f"Columnas del CSV: {df_input.columns.tolist()}")

        # Validación de columnas
        missing_cols = [col for col in FEATURE_COLUMNS if col not in df_input.columns]

        print(f"Columnas faltantes: {missing_cols}")

        if missing_cols:
            return f"Error: El CSV debe tener 11 columnas. Faltan: {', '.join(missing_cols)}", pd.DataFrame()

        # Registrar el archivo de entrada original como artefacto
        mlflow.log_artifact(csv_file.name) 

        df_predict = df_input[FEATURE_COLUMNS]
        df_predict = df_predict.astype('float64')
        predictions = loaded_model.predict(df_predict)

        df_output = df_input.copy()
        df_output['Predicted_Quality'] = predictions
        
        # Registrar métricas clave del lote
        avg_quality = df_output['Predicted_Quality'].mean()
        mlflow.log_metric("batch_size", len(df_output))
        mlflow.log_metric("avg_predicted_quality", avg_quality)

        # Generar Resumen Gen AI
        summary_prompt = f"""
        Genera un breve resumen (máx. 30 palabras) de la calidad de los {len(df_output)} vinos analizados. 
        Calidad promedio: {avg_quality:.2f}. Menciona la tendencia general (alta, media o baja) del lote.
        """
        try:
            gemini_model = get_gemini_client()
            response = gemini_model.generate_content(summary_prompt)
            summary_text = response.text.strip()
            summary = f"Resumen Gen AI:\n{summary_text}"
            mlflow.log_text(summary_text, "batch_summary.txt") # Registrar resumen
        except Exception:
            summary = f"Resumen de Calidad Promedio: {avg_quality:.2f}. (Error Gen AI)"

        # Opcional: Registrar el CSV de salida con las predicciones
        output_buffer = io.StringIO()
        df_output.to_csv(output_buffer, index=False)
        mlflow.log_text(output_buffer.getvalue(), "batch_predictions_output.csv")

        return summary, df_output


# --- Construcción de la Interfaz Gradio (usando Blocks) ---
# ... (El resto del código de Gradio se mantiene igual, usando las nuevas funciones)

with gr.Blocks(
    title="Predicción y Generación de Insights sobre Calidad de Vinos con MLOps",
    theme=gr.themes.Soft(),
) as app:
    gr.Markdown(
        f"""
        # Predicción y Generación de Insights sobre Calidad de Vinos con MLOps
        **Modelo Actual:** `{MODEL_STATUS}`.
        """
    )

    with gr.Tab("Predicción Individual"):
        # Componentes de entrada dinámicos (11 números)
        input_components_single = []
        for feature in FEATURE_COLUMNS:
            default_value = INITIAL_VALUES.get(feature, 0.5) 
            input_components_single.append(
                gr.Number(
                    label=f"{feature} (g/dm³ o valor correspondiente)", 
                    value=default_value
                )
            )

        # Botón y outputs
        predict_btn_single = gr.Button("Predecir Calidad", variant="primary")
        output_components_single = [
            gr.Textbox(label="Resultado de la Predicción", key="prediction_output"),
            gr.Textbox(label="Insight de Calidad", key="insight_output"),
            gr.Textbox(label="Explicación Gen AI (Sommelier Virtual)", key="genai_output")
        ]
        
        # Conexión de la función al botón
        predict_btn_single.click(
            fn=predict_wine_quality,
            inputs=input_components_single,
            outputs=output_components_single
        )

    with gr.Tab("Predicción por Lote (CSV)"):
        gr.Markdown(
            """
            Sube un archivo **CSV** que contenga las 11 columnas de características. 
            El modelo ejecutará la predicción en todas las filas y devolverá la calidad predicha (`Predicted_Quality`).
            """
        )
        
        # Entrada CSV
        csv_input = gr.File(
            label="Cargar Archivo CSV", 
            file_types=[".csv"]
        )

        # Salidas CSV
        summary_output = gr.Textbox(label="Resumen de Lote Gen AI", interactive=False)
        dataframe_output = gr.Dataframe(
            label="Tabla de Resultados", 
            headers=FEATURE_COLUMNS + ['Predicted_Quality'],
            interactive=False
        )

        # Botón y Conexión de la función
        predict_btn_batch = gr.Button("Predecir Lote CSV", variant="primary")
        
        predict_btn_batch.click(
            fn=predict_from_csv,
            inputs=csv_input,
            outputs=[summary_output, dataframe_output]
        )

# Ejecutar la aplicación
if __name__ == "__main__":
    app.launch(inbrowser=True, show_api=False)