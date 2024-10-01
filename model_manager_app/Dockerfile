# Use uma imagem base oficial do Python
FROM python:3.9-slim

# Defina o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copie o arquivo de requisitos para instalar dependências
COPY requirements.txt requirements.txt

# Instale as dependências
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie os arquivos da API para o container
COPY app.py /app/app.py
COPY templates /app/templates
COPY cluster_analysis.csv /app/cluster_analysis.csv
COPY modelo_LogisticRegression.pkl /app/modelo_LogisticRegression.pkl
COPY preprocessor_Log.pkl /app/preprocessor_Log.pkl
COPY modelo_kmeans.pkl /app/modelo_kmeans.pkl
COPY preprocessor.pkl /app/preprocessor.pkl
COPY models.py /app/models.py
COPY pipeline_completo.pkl /app/pipeline_completo.pkl

# Exporte a porta 80
EXPOSE 80

# Comando para rodar a aplicação
CMD ["gunicorn", "--workers", "3", "--bind", "0.0.0.0:80", "app:app"]
