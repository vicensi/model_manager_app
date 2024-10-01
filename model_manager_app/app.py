from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
from datetime import datetime
import joblib
from models import run_model1, run_model2
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


app = Flask(__name__)


# Log function
def log_entry_exit(model_name, inputs, output):
    log_data = {
        "model_name": model_name,
        "timestamp": datetime.now(),
        "inputs": inputs,
        "output": output
    }
    df = pd.DataFrame([log_data])
    df.to_csv('logs.csv', mode='a', header=False, index=False)

# Home page to select model
@app.route('/')
def index():
    return render_template('index.html')

# Route for Model 1
@app.route('/model1', methods=['GET', 'POST'])
def model1():
    if request.method == 'POST':
        inputs = request.form.to_dict()
        cluster, persona_label, credit_probability = run_model1(inputs)
        output = {
            'group': int(cluster),
            'persona_label': persona_label,
            'credit_probability': credit_probability
        }
        # Armazena logs
        log_entry_exit('Model 1', inputs, output)
        # Retorna saída em formato JSON
        return f"""
        <h1>Resultado da Previsão - Model 1</h1>
        <pre>{output}</pre>
        <br>
        <form action="/model1" method="GET">
            <button type="submit">Fazer outra previsão</button>
        </form>
        <br>
        <form action="/" method="GET">
            <button type="submit">Voltar para seleção de modelos</button>
        </form>
        """
    return render_template('model1_form.html')

# Route for Model 2
@app.route('/model2', methods=['GET', 'POST'])
def model2():
    if request.method == 'POST':
        inputs = request.form.to_dict()
        output = run_model2(inputs)
        log_entry_exit('Model 2', inputs, output)
        return f"""
        <h1>Resultado da Previsão - Model 2</h1>
        <pre>{output}</pre>
        <br>
        <form action="/model2" method="GET">
            <button type="submit">Fazer outra previsão</button>
        </form>
        <br>
        <form action="/" method="GET">
            <button type="submit">Voltar para seleção de modelos</button>
        </form>
        """
    return render_template('model2_form.html')

if __name__ == '__main__':
    app.run(debug=True)
