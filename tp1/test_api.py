import requests

BASE_URL = "http://localhost:8000"

def test_predict():
    payload = {
        "data": [
            [5.1, 3.5, 1.4, 0.2],
            [6.2, 3.4, 5.4, 2.3]
        ]
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    if response.status_code == 200:
        print("Predict endpoint works! Output:", response.json())
    else:
        print("Predict endpoint failed:", response.text)

def test_update_model():
    payload = {"model_uri": "models:/tracking-quickstart/2"}  # send as JSON object
    response = requests.post(f"{BASE_URL}/update-model", json=payload)
    if response.status_code == 200:
        print("Update-model endpoint works! Output:", response.json())
    else:
        print("Update-model endpoint failed:", response.text)

if __name__ == "__main__":
    test_predict()
    test_update_model()
    print("All tests completed.")
