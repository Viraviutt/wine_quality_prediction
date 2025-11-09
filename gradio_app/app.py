import gradio as gr
import pandas as pd
import numpy as np
import mlflow.pyfunc
import os
from openai import OpenAI
from dotenv import load_dotenv

# Carga variables de entorno (para la clave de OpenAI)
load_dotenv()

# --- Configuración y Carga del Modelo ---

# Asegúrate de que este nombre coincida con el registrado en train.py
MODEL_NAME = "wine_quality_model" 
# Carga la versión 'Production' del modelo registrado en MLflow
# NOTA: Debes tener el servidor de MLflow (mlflow ui) activo o haber configurado
# el tracking remoto para que esto funcione.
try:
    # Intenta cargar el modelo de 'Production'
    logged_model = f'models:/{MODEL_NAME}/Production'
    model = mlflow.pyfunc.load_model(logged_model)
    MODEL_STATUS = "Modelo Productivo (MLflow Registry)"
except Exception as e:
    # Fallback si MLflow no está disponible o el modelo no está en Production
    print(f"Error al cargar modelo de MLflow Registry: {e}")
    MODEL_STATUS = "Error al cargar modelo. Verifique MLflow UI."
    # Define un modelo dummy para que la app pueda iniciar
    class DummyModel:
        def predict(self, df):
            # Retorna una predicción base si el modelo real falla
            return np.array([6.0])
    model = DummyModel()


# --- Configuración de la Interfaz ---

# Columnas de entrada para la predicción
FEATURE_COLUMNS = [
    "fixed acidity", "volatile acidity", "citric acid", 
    "residual sugar", "chlorides", "free sulfur dioxide", 
    "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
]

# Valores iniciales de ejemplo (típicos de un vino blanco)
INITIAL_VALUES = {
    "fixed acidity": 7.0, 
    "volatile acidity": 0.27, 
    "citric acid": 0.36, 
    "residual sugar": 20.7, 
    "chlorides": 0.045, 
    "free sulfur dioxide": 45.0, 
    "total sulfur dioxide": 170.0, 
    "density": 1.001, 
    "pH": 3.0, 
    "sulphates": 0.45, 
    "alcohol": 8.8
}


# --- Funciones Lógicas de Gradio ---

def get_gen_ai_explanation(input_data, prediction):
    """Genera una explicación Gen AI para la predicción."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Crear una descripción concisa de las características del vino
    characteristics = ", ".join([
        f"{col}: {val}" for col, val in zip(FEATURE_COLUMNS, input_data.iloc[0])
    ])
    
    prompt = f"""
    Un modelo de Machine Learning predijo que la calidad de un vino blanco es de {prediction[0]:.2f} (escala 0-10).
    Las propiedades fisicoquímicas del vino son: {characteristics}. 

    Genera una explicación de una sola frase (máx. 25 palabras) que justifique por qué la calidad es alta, baja o media,
    basándote en las propiedades dadas. Enfócate en la relación entre alcohol/acidez/azúcar y calidad.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un sommelier experto en química del vino y explicas predicciones de calidad."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=80
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error Gen AI: Verifique la API Key de OpenAI. {e}"


def predict_wine_quality(*args):
    """Función de predicción que acepta los 11 argumentos de entrada."""
    
    # Crea el DataFrame de entrada con los 11 valores
    input_values = list(args)
    input_df = pd.DataFrame([input_values], columns=FEATURE_COLUMNS)
    
    # 1. Ejecutar la Predicción
    prediction = model.predict(input_df)
    quality_score = prediction[0]

    # 2. Generar Explicación Gen AI
    explanation = get_gen_ai_explanation(input_df, prediction)
    
    # 3. Formatear la predicción para mostrar
    formatted_prediction = f"Predicción de Calidad: {quality_score:.2f} / 10"

    # 4. Determinar si el vino es "Bueno" o "Malo" para un insight adicional
    if quality_score >= 7.0:
        insight = "¡Excelente Predicción! Probablemente un vino de alta calidad."
    elif quality_score >= 5.0:
        insight = "Calidad promedio. El modelo sugiere un vino bebible."
    else:
        insight = "Baja calidad. Se recomienda precaución."
    
    # Retorna todos los outputs para los componentes de Gradio
    return formatted_prediction, insight, explanation


# --- Construcción de la Interfaz Gradio ---

# Componentes de entrada dinámicos
input_components = []
for feature in FEATURE_COLUMNS:
    # Usar el valor inicial del diccionario
    default_value = INITIAL_VALUES.get(feature, 0.5) 
    input_components.append(
        gr.Number(
            label=f"{feature} (g/dm³ o valor correspondiente)", 
            value=default_value
        )
    )

# Componentes de salida
output_components = [
    gr.Textbox(label="📊 Resultado de la Predicción", key="prediction_output"),
    gr.Textbox(label="🍷 Insight de Calidad", key="insight_output"),
    gr.Textbox(label="💡 Explicación Gen AI (Sommelier Virtual)", key="genai_output")
]

# Interfaz principal
iface = gr.Interface(
    fn=predict_wine_quality,
    inputs=input_components,
    outputs=output_components,
    title="🍾 Predicción y Generación de Insights sobre Calidad de Vinos con MLOps",
    description=f"""
    Introduce las 11 propiedades fisicoquímicas del vino para predecir su calidad (escala 0-10).
    El modelo actual es la versión **Production** cargada desde el **MLflow Model Registry** ({MODEL_STATUS}).
    La explicación textual se genera automáticamente por un LLM (Gen AI) para interpretar el resultado.
    """,
    live=False,
    allow_flagging="never"
)

with gr.Blocks() as app:
    gr.Markdown("## 🍾 Predicción y Generación de Insights sobre Calidad de Vinos con MLOps")
    

# Ejecutar la aplicación
if __name__ == "__main__":
    iface.launch(inbrowser=True, show_api=False)
