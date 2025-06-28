FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/checkpoints
COPY src/checkpoints/best.pth.tar /app/checkpoints/

ENTRYPOINT ["python3", "-u", "src/engine.py"]