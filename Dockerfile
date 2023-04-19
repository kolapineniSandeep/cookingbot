FROM python:3.10-slim

WORKDIR /AI_BOT
COPY requirements.txt .
RUN apt-get update && apt-get install -y build-essential && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get remove -y build-essential && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY . .

ENV FLASK_ENV=production
EXPOSE 5001
CMD ["gunicorn", "-b", "0.0.0.0:5001", "app:app"]
