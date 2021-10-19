import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'N':1,'P':1,'K':1,'ph':1,'rainfall':1})

print(r.json())