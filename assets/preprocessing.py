from sklearn.preprocessing import LabelEncoder
import pandas as pd

def preprocess_data(df):
    df = df.copy()

    # Eliminar columnas irrelevantes
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Convertir target si está presente
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Convertir columnas tipo object a categóricas codificadas
    object_cols = df.select_dtypes(include='object').columns

    for col in object_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # TotalCharges puede venir como string si hay errores de carga
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(0, inplace=True)

    return df
