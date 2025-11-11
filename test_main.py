from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_infer_predict_positive_en():
    response = client.post("/predict/",
                           json={"text": "I like this film! Very interesting!"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'POSITIVE'


def test_infer_predict_negative_en():
    response = client.post("/predict/",
                           json={"text": "I don't like this film! Very boring!"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'NEGATIVE'


def test_infer_predict_positive_ru():
    response = client.post("/predict/",
                           json={"text": "Мне понравилось! Всем советую к просмотру"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'POSITIVE'


def test_infer_predict_negative_ru():
    response = client.post("/predict/",
                           json={"text": "Мне очень не понравилось, какой-то ужас"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'NEGATIVE'


def test_infer_predict_neutral_ru():
    response = client.post("/predict/",
                           json={"text": "В целом сойдет, посмотреть можно.."})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'NEUTRAL'
