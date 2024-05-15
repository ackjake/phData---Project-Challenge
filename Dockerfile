FROM python:3.9

RUN apt-get update && apt-get install -y \
  nginx
  && rm -rf /var/lib/apt/lists/*

COPY serve .
COPY model/model.pkl .

RUN serve
