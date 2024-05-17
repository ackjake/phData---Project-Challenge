docker build -t phdata-app .
docker images
docker run -dp 9999:9999 --cpus=1 --memory=256m phdata-app

curl -X POST "http://0.0.0.0:9999/predict/" -H "accept: application/json" \
    -H "Content-Type: application/json" -d \
    '{"zipcode": "98075",
        "features": {"bedrooms": 4,
        "bathrooms": 2.5,
        "sqft_living": 2550,
        "sqft_lot": 4630,
        "floors": 2.0,
        "sqft_above": 2550,
        "sqft_basement": 0} 
  }'
