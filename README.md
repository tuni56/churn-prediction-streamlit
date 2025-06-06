# churn-prediction-streamlit
# 📊 Customer Churn Prediction Dashboard

Este proyecto presenta una solución interactiva desarrollada con **Streamlit** para predecir la fuga de clientes (churn) en una empresa de telecomunicaciones. Utiliza un modelo de machine learning entrenado con **XGBoost** y proporciona explicaciones visuales con **SHAP** para facilitar la toma de decisiones.

---

## 🚀 Características

- Dashboard profesional y 100% visual
- Formulario para evaluar clientes individuales
- Visualización de importancia de variables con SHAP
- Métricas ejecutivas para decisiones estratégicas
- HTML personalizado y marca personal

---

## 🛠️ Stack Tecnológico

- Python 3.10+
- Streamlit
- pandas
- scikit-learn
- XGBoost
- SHAP
- matplotlib
- joblib

---

## 🧪 Estructura del Proyecto

```
churn-prediction-streamlit/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── Telco-Customer-Churn.csv
├── models/
│   └── xgb_churn_model.joblib
├── utils/
│   ├── preprocessing.py
│   └── shap_helper.py
└── assets/
    └── logo.png
```

---

## ▶️ Cómo correr el proyecto

```bash
# 1. Crear y activar entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar la app
streamlit run app.py
```

---

## 📈 Impacto Empresarial

Esta solución permite a los equipos ejecutivos:

- Identificar segmentos de clientes con alto riesgo de churn
- Interpretar decisiones del modelo con transparencia
- Tomar acciones personalizadas basadas en datos

---

## 📩 Contacto

Desarrollado por **Rocio**, Data Scientist & AWS Solutions Architect.

© 2025 Rocio | Customer Churn Dashboard
