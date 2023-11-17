import streamlit as st
import pandas as pd
import numpy as np
from joblib import  load
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Cargar el modelo (ajustar la ruta según sea necesario)
rf_clf_model = load('rf_clf.joblib')

def heuristic_prediction(xs, rf_clf_model, delay_paid_bill=31, delay_non_paid_bill=100):
  clf_preds = rf_clf_model.predict(xs)
  hr_preds = np.where(clf_preds == 1, delay_paid_bill, delay_non_paid_bill)
  return hr_preds


st.title("Predicción de demoras de Cuentas por Cobrar")


# Subir archivo
uploaded_file = st.file_uploader("Subir archivo Excel/CSV", type=['csv', 'xlsx'])
if uploaded_file is not None:
    # Leer el archivo
    if uploaded_file.name.endswith('.csv'):
        df_org = pd.read_csv(uploaded_file)
    else:
        df_org = pd.read_excel(uploaded_file)

    # Preprocessing steps
    # Select only the required columns
    df = df_org[['Fecha_Emision', 'Fecha_Vencimiento', 'Desc_CondicionPago', 'ImporteTotal', 'DescTipo', 'Pagada?']].copy()

    # Handle missing values for 'ImporteTotal' using median imputation
    imputer = SimpleImputer(strategy='median')
    df['ImporteTotal'] = df['ImporteTotal'].str.replace(',', '.').astype(float)
    
    if 'Será Pagada?' not in df.columns:
        df['ImporteTotal'] = imputer.fit_transform(df[['ImporteTotal']])

    # Feature Engineering
    # Extract features from 'Fecha_Emision' and 'Fecha_Vencimiento'
    df['Days_to_Pay'] = (df['Fecha_Vencimiento'] - df['Fecha_Emision']).dt.days

    # Convert categorical variables to string type for encoding
    df['Desc_CondicionPago'] = df['Desc_CondicionPago'].astype(str)
    df['DescTipo'] = df['DescTipo'].astype(str)
    
    if 'Será Pagada?' not in df.columns:
        preds = rf_clf_model.predict(df)
        preds= rf_clf_model.predict(df)
        delay = heuristic_prediction(df, rf_clf_model)

    # Agregar columnas requeridas
    df['Será Pagada?'] = preds
    df['Demora Estimada'] = delay
    df['Fecha de Pago real estimada'] = df['Fecha_Emision'] + pd.to_timedelta(df['Demora Estimada'], unit='D')

    # Mostrar DataFrame
    st.write(df)

   
