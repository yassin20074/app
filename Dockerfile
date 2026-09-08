FROM python:3.11-slim

# منع ببيثون من كتابة ملفات bytecode وتفعيل الـ Logging المباشر
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# تثبيت مكتبات النظام الرسومية المطلوبة لـ OpenCV و MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# تثبيت المكتبات أولاً للاستفادة من الـ Docker Caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي ملفات المشروع
COPY main.py .

EXPOSE 8000

# قراءة متغيّر المنفذ PORT تلقائياً من Railway
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
