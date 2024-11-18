FROM python:3.8-slim

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código y datos
COPY . .

# CMD ["python", "main.py"]
CMD ["sh", "-c", "python automl_pipeline.py && python main.py"]

