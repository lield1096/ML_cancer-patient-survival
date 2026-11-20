import pytest
from fastapi.testclient import TestClient

from survival_predictor import api

PATIENT = {
    "age": 55,
    "tumor_size": 30,
    "regional_nodes_examined": 14,
    "regional_nodes_positive": 3,
}


@pytest.fixture
def client(trained_model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(trained_model_path))
    with TestClient(api.app) as c:
        yield c


@pytest.fixture
def client_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    body = client.get("/model").json()
    assert body["features"] == list(PATIENT)
    assert "roc_auc" in body["test_metrics"]


def test_predict_single(client):
    r = client.post("/predict", json=PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability_dead"] <= 1
    assert body["predicted_status"] in {"Alive", "Dead"}


def test_predict_batch(client):
    r = client.post("/predict/batch", json={"patients": [PATIENT, PATIENT | {"tumor_size": 90}]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


@pytest.mark.parametrize(
    "patch",
    [
        {"age": -5},
        {"tumor_size": "big"},
        {"regional_nodes_positive": 20},  # more positive than examined
    ],
)
def test_invalid_input_returns_422(client, patch):
    assert client.post("/predict", json=PATIENT | patch).status_code == 422


def test_missing_field_returns_422(client):
    body = {k: v for k, v in PATIENT.items() if k != "age"}
    assert client.post("/predict", json=body).status_code == 422


def test_empty_batch_rejected(client):
    assert client.post("/predict/batch", json={"patients": []}).status_code == 422


def test_no_model_returns_503(client_without_model):
    assert client_without_model.get("/health").json()["model_loaded"] is False
    assert client_without_model.post("/predict", json=PATIENT).status_code == 503
