import requests
import json

url = 'http://127.0.0.1:5000/predict'

input_data = {
    "features": [5.1, 3.5, 1.4, 0.2]  
}


response = requests.post(url, json=input_data)


if response.status_code == 200:
    print('Prediction:', response.json())
else:
    print('Failed to get a valid response from the API. Status code:', response.status_code)
    print('Response:', response.text)
