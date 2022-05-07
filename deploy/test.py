import requests

resp = requests.post(
"http://localhost:5000/predict",
files={'file': open('subset_dataset/Albrecht Durer/23.jpg', 'rb')}
)

# resp = requests.post(
#     "https://image-classifier-flask-demo.herokuapp.com/predict",
#     files={'file': open('cat.jpg', 'rb')}
# )

# https://image-classifier-flask-demo.herokuapp.com/

print(resp.text)
