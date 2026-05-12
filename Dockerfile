
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app


RUN pip install --no-cache-dir \
    transformers \
    requests \
    python-dotenv


COPY attention.py block.py data_preparation.py dataset.py \
     eval.py feedforward.py gpt.py train.py main.py ./

COPY input.txt ./

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

VOLUME ["/app/output"]

# Pass your .env at runtime:  docker run --gpus all --env-file .env ...
CMD ["python", "main.py"]