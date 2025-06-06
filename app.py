
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from utils.preprocessing import preprocess_data
from utils.shap_helper import plot_shap_beeswarm

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

# Título y descripción
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("""
Este dashboard permite explorar los factores que influyen en la fuga de clientes para una compañía de telecomunicaciones.
Carga el dataset, visualiza las variables importantes, y evaluá la probabilidad de que un cliente se dé de baja.
""")

# Carga de datos
@st.cache_data

def load_data():
    df = pd.read_csv("data/Telco-Customer-Churn.csv")
    return df

data = load_data()

# Preprocesamiento
X = preprocess_data(data.drop("Churn", axis=1))
y = data["Churn"].map({'Yes': 1, 'No': 0})

# Cargar modelo entrenado
model = joblib.load("models/xgb_churn_model.joblib")

# Sección 1: Visualización EDA
st.header("🔍 Exploración de Datos")
st.dataframe(data.head())

# Sección 2: Importancia de Variables
st.header("📌 Importancia de Variables (SHAP)")
st.markdown("Estas visualizaciones muestran qué variables más influyen en las predicciones de churn.")

fig = plot_shap_beeswarm(model, X)
st.pyplot(fig)

# Sección 3: Predicción de Cliente Individual
st.header("🧪 Evaluación de Cliente Individual")
st.markdown("Completá el formulario para evaluar el riesgo de churn de un cliente específico.")

with st.form("customer_form"):
    input_data = {}
    for col in X.columns:
        if X[col].nunique() <= 5:
            input_data[col] = st.selectbox(col, options=sorted(data[col].unique()))
        else:
            input_data[col] = st.number_input(col, value=float(X[col].mean()))
    submitted = st.form_submit_button("Predecir")

if submitted:
    df_input = pd.DataFrame([input_data])
    df_input_processed = preprocess_data(df_input)
    prediction = model.predict_proba(df_input_processed)[0][1]
    st.metric(label="Probabilidad de Churn", value=f"{prediction:.2%}")

    explainer = shap.Explainer(model)
    shap_values = explainer(df_input_processed)

    st.subheader("🔍 Interpretación SHAP para este cliente")
    st.set_option("deprecation.showPyplotGlobalUse", False)
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(bbox_inches='tight')

# Footer HTML personalizado
st.markdown("""
<hr style="border:1px solid #eee" />
<div style='text-align: center; font-size: small'>© 2025 Rocio | Customer Churn Dashboard</div>
""", unsafe_allow_html=True)
