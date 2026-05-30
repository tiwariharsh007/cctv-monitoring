FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && pip install --upgrade pip \
    && pip install -r requirements.txt

CMD ["python", "app.py"]