FROM python:3.9

RUN apt-get update && apt-get install -y \
  nginx
  && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

COPY serve .
COPY model/model.pkl .

CMD ["uvicorn", "serve:app", "--reload", "--workers", "1", "--host", "0.0.0.0", "--port", "9999"]
