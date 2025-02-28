FROM python:3.10-slim

WORKDIR /app

# Instalar dependências do sistema necessárias para o ChromaDB
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements.txt
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código da aplicação
COPY . .

# Criar diretórios necessários
RUN mkdir -p chroma_db rags

# Expor a porta do Streamlit
EXPOSE 8501

# Desativar o file watcher para evitar problemas
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE="none"

# Comando para executar o Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]