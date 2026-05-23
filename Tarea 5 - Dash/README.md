# Dashboard Predictivo Saber 11 - Atlántico

Este tablero integra tres modelos:

1. Modelo familiar: clasificación de alto desempeño.
2. Modelo colegio: regresión del puntaje global con variables institucionales.
3. Modelo general: regresión del puntaje global con variables mixtas.

## Archivos necesarios

Coloca el archivo limpio como:

```bash
saber11_limpio.csv
```

o define la ruta con:

```bash
export DATA_PATH="ruta/al/archivo.csv"
```

## Entrenar modelos

```bash
python train_all_models.py
```

Esto crea la carpeta `models/` con los artefactos necesarios.

## Ejecutar tablero

```bash
python app.py
```

Abrir:

```text
http://localhost:8051
```

## Docker

```bash
docker build -t saber11-dashboard .
docker run -p 8051:8051 saber11-dashboard
```
