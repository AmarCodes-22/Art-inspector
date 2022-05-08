import requests
from pprint import pprint

resp = requests.post(
    "http://localhost:5000/predict",
    files={'file': open('subset_dataset/Albrecht Durer/23.jpg', 'rb')}
)

# https://art-inspector.herokuapp.com/
# resp = requests.post(
#     "https://art-inspector.herokuapp.com/predict",
#     files={'file': open('subset_dataset/Albrecht Durer/23.jpg', 'rb')}
# )

# resp = requests.post(
#     "https://image-classifier-flask-demo.herokuapp.com/predict",
#     files={'file': open('cat.jpg', 'rb')}
# )

# https://image-classifier-flask-demo.herokuapp.com/

pprint(resp.text)
