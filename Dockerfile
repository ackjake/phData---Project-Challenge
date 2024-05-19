FROM python:3.9

WORKDIR /usr/src/app

COPY serve.py .
COPY model/model.pkl .
COPY model/model_features.json .
COPY data/zipcode_demographics.csv .
COPY requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 9999
CMD ["uvicorn", "serve:app", "--workers", "1", "--host", "0.0.0.0", "--port", "9999"]
