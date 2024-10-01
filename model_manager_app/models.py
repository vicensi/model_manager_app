# Simulações simples dos modelos
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from flask import jsonify
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Carregar o modelo e o pré-processador
kmeans_model = joblib.load('modelo_kmeans.pkl')
preprocessor = joblib.load('preprocessor.pkl')
LogisticRegression_model = joblib.load('modelo_LogisticRegression.pkl')
modelo_LogisticRegression = joblib.load('modelo_LogisticRegression.pkl')
preprocessor_log = joblib.load('preprocessor_Log.pkl')
# Carregar o pipeline completo
pipeline_completo = joblib.load('pipeline_completo.pkl')


# Carregar a análise de clusters previamente calculada
cluster_analysis = pd.read_csv('cluster_analysis.csv')  


def run_model1(inputs):
    # Clustering
    # Preprocessar os dados de entrada
    input_df = pd.DataFrame([inputs])

    # Definir as colunas categóricas e numéricas
    categorical_columns = ['Gender', 'approv_in_adv', 'loan_type', 'loan_purpose',
                        'Neg_ammortization',
                         'occupancy_type',
                        'credit_type', 'co_applicant_credit_type', 'age',
                       'submission_of_application', 'Region']
    numeric_columns = ['loan_amount', 'term', 'property_value', 'income', 'Credit_Score']
    
    # Preprocessar os dados de entrada
    processed_data = preprocessor.transform(input_df)

    # Prever o cluster
    cluster = kmeans_model.predict(processed_data)[0]
    
    # Obter o label da persona e a probabilidade de fraude correspondentes
    persona_label = cluster_analysis.loc[cluster, 'persona_label']
    credit_probability = cluster_analysis.loc[cluster, 'Status']
    
    return cluster, persona_label, credit_probability


def run_model2(inputs):
    # Clustering

    # Preprocessar os dados de entrada
    input_df = pd.DataFrame([inputs])

    # Definir as colunas categóricas e numéricas
    categorical_columns = ['approv_in_adv', 'loan_type', 'loan_purpose',
                        'Neg_ammortization',
                         'occupancy_type',
                        'credit_type', 'co_applicant_credit_type', 'age',
                       'submission_of_application']
    numeric_columns = ['loan_amount', 'term', 'property_value', 'income', 'Credit_Score']
    
    # Fazer a previsão com o modelo carregado
    previsao = pipeline_completo.predict(input_df)[0]

         
    return previsao
