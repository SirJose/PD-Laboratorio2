# Version 1.0.0

Version 1.0.0 del Laboratorio 2 de Product Development.

# Instrucciones
Para ejecutar este proyecto se deben ejecutar los siguientes comandos:

1. Construir una imagen Docker
```bash
docker build -t automl-dockerizer:latest . 
```
#### Modo batch prediction
2. Para modo batch prediction. Puede tomar algunos minutos en completar su ejecución dependiendo de la computadora:
```bash
docker run --env-file .env -v "$(pwd)/data:/data" automl-dockerizer:latest
```
#### Modo API
3. Para modo API
```bash
docker run --env-file .env -p 8000:8000 -v "$(pwd)/data:/data" automl-dockerizer:latest
```

# Nota importante
> **Note**: En los comandos se utiliza _pwd_ por simplicidad. Sin embargo, puede ocasionar errores. Si ese es el caso favor de reemplazar _pwd_ por la ruta. Por ejemplo:

2. Para modo batch prediction
```bash
docker run --env-file .env -v "C:\Users\josee\OneDrive\Documents\Galileo\Maestria\4to Trimestre\Product Development\Laboratorio2\data:/data" automl-dockerizer:latest
```
3. Para modo API
```bash
docker run --env-file .env -p 8000:8000 -v "$(pwd)/data:/data" automl-dockerizer:latest
```

