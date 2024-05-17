curl -X POST "http://localhost:8008/predict/" -H "accept: application/json" \
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
