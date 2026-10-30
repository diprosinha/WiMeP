# WiMeP web server (CPU). Build:  docker build -t wimep .
# Run:  docker run -p 8080:8080 wimep      (set -e PORT=... to change the port)
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .
USER user
# models are bundled; the transformers library never needs the network
ENV PORT=8080 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ['PORT'])"
CMD ["sh", "-c", "exec python wimep.py serve --host 0.0.0.0 --port ${PORT}"]
