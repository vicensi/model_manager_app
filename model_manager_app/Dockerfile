# Use uma imagem base oficial do Python
FROM python:3.8-slim-buster

# Defina o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copie o arquivo de requisitos para instalar dependências
COPY requirements.txt requirements.txt

# Instale as dependências
RUN pip install -r requirements.txt

# Copie todo o conteúdo do projeto para o contêiner
COPY . .

# Exporte a porta 80
EXPOSE 80

# Comando para rodar a aplicação
CMD ["gunicorn", "--workers", "3", "--bind", "0.0.0.0:80", "app:app"]