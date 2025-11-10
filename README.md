# 🍾 Predicción y Generación de Insights sobre Calidad de Vinos con MLOps

Este proyecto implementa un flujo de MLOps de principio a fin para predecir la calidad de vinos (escala 0-10) utilizando un modelo de Machine Learning (Random Forest) y sirve las predicciones a través de una interfaz web con Gradio. El proyecto utiliza **MLflow** para el seguimiento de experimentos, el registro de modelos y el control del ciclo de vida.

## ✨ Características Principales de la Aplicación

| Característica | Herramienta | Descripción |
| :--- | :--- | :--- |
| **Seguimiento MLOps** | MLflow | **Registra cada predicción** realizada por la interfaz Gradio como un nuevo *Run*, guardando los *inputs* y *outputs* para trazabilidad. |
| **Model Registry** | MLflow | Carga la versión del modelo de mayor calidad (etiquetada como `status: production`) directamente desde el Registro de Modelos. |
| **Predicción Individual** | Gradio | Interfaz con 11 campos numéricos para ingresar manualmente las propiedades fisicoquímicas del vino. |
| **Predicción por Lote** | Gradio | Permite subir un archivo CSV para obtener predicciones en masa, registrando el archivo y los resultados en MLflow. |
| **Explicaciones Gen AI** | Gemini API | Utiliza un Large Language Model (LLM) para generar una explicación de texto (rol de "Sommelier Virtual") que justifica la predicción del modelo de ML. |

## ⚙️ Pre-requisitos

Para instalar y ejecutar este proyecto, necesitas tener instalado:

* **Python 3.11.14**
* **Conda/Mamba** (Recomendado para la gestión de entornos)
* Una clave de la **Gemini API**.

## 🚀 Guía de Instalación y Ejecución

Sigue estos pasos detallados para configurar y lanzar la aplicación en una nueva máquina.

### Paso 1: Clonar el Repositorio y Configurar el Entorno

```bash
# 1. Clonar tu repositorio (reemplaza con tu URL real si aplica)
git clone https://github.com/Viraviutt/wine_quality_prediction
cd wine_quality_prediction

# 2. Crear y activar el entorno virtual con Conda
conda env create -f conda.yaml
conda activate wine-mlops-env

# 3. Instalar las dependencias
pip install -r requirements.txt 
# Si no tienes requirements.txt, instala las librerías principales:
# pip install mlflow scikit-learn pandas numpy gradio python-dotenv openai google-generativeai
```
### Paso 2: Configurar las Variables de Entorno y el Backend

La aplicación requiere la clave de Gemini para las explicaciones y una base de datos SQLite para MLflow.

```bash
# 1. Configurar la API KEY
GEMINI_API_KEY=<tu_api_key>

# 2. Configurar la Base de Datos de MLflow
touch mlflow.db
conda activate wine-mlops-env
```

### 3. Iniciar el Servidor de MLflow
El servidor de MLflow debe estar activo para que la aplicación Gradio pueda cargar el modelo y hacer el logging de las predicciones a través de la API HTTP.

```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000
```

### 4. Entrenamiento y Registro del Modelo
Debes ejecutar el script de entrenamiento para generar el modelo y registrarlo en la versión status: stagging dentro de la base de datos mlflow.db.

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
mlflow run project
```
*NOTA*: Deja esta terminal ejecutando el servidor y abre una nueva terminal para el siguiente paso.

### Paso 5: Ejecutar la Aplicación Gradio
En la nueva terminal (con el entorno wine-mlops-env activado), ejecuta la aplicación web. El script app.py se conectará a http://localhost:5000 para cargar el modelo en producción.

```bash
# Asegúrate de estar en el directorio correcto y el entorno activado
python app.py
```

Tu navegador se abrirá automáticamente en la dirección http://127.0.0.1:7860. Cada interacción en Gradio (predicción individual o lote) ahora se registrará como un nuevo Run en tu servidor de MLflow.