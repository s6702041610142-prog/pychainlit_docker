FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces (Docker SDK) expects the app on port 7860.
# Render/Railway/Fly inject their own $PORT — this falls back to 7860 if unset.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "chainlit run cl_app.py --host 0.0.0.0 --port ${PORT} --headless"]
