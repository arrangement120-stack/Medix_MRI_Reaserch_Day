FROM python:3.11-slim

WORKDIR /app

# مكتبات نظام خفيفة لازمة لـ PyTorch و HDF5 و Matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libhdf5-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# تثبيت المكتبات و fastMRI وقت البناء للاستفادة من Docker Cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FASTMRI=0.3.0 \
    VCS_VERSIONING_PRETEND_VERSION_FOR_FASTMRI=0.3.0 \
    pip install --no-cache-dir --no-deps \
    https://github.com/facebookresearch/fastMRI/archive/refs/heads/main.zip

# نسخ ملفات التطبيق والـ CLI
COPY app.py medix_engine.py medix_cli.py ./

# إعدادات التشغيل داخل الحاوية (محليًا بدون حد لحجم الملفات)
ENV GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT="7860" \
    GRADIO_SHARE="false" \
    PYTHONUNBUFFERED="1"

EXPOSE 7860

CMD ["python", "app.py"]
