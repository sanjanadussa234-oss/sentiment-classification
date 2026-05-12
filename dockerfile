FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --default-timeout=200 --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]