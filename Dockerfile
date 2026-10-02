# ── CoughAI · Cloud Run ──
FROM python:3.11-slim

# ระบบ lib ที่ librosa / soundfile ต้องใช้
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run ส่ง PORT=8080 มาให้
ENV PORT=8080

# 1 worker / หลาย thread → โมเดลโหลดครั้งเดียวแชร์ใน RAM
# timeout 0 = ไม่ตัด request ระหว่างโหลดโมเดล/ประมวลผลเสียง
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
