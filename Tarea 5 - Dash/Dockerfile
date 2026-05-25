FROM python:3.11-slim

WORKDIR /app

COPY ["Tarea 5 - Dash/requirements.txt", "/app/requirements.txt"]

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/

EXPOSE 8050

CMD ["python", "Tarea 5 - Dash/app.py"]