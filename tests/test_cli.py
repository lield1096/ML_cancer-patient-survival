import pandas as pd
import pytest

from survival_predictor.cli import main
from survival_predictor.data import DataValidationError
from survival_predictor.predict import SurvivalModel


def test_cli_train_then_predict(config_file, tmp_path, capsys):
    assert main(["train", "--config", str(config_file)]) == 0
    assert '"best_model"' in capsys.readouterr().out

    inp = tmp_path / "patients.csv"
    pd.DataFrame(
        [{"age": 50, "tumor_size": 20, "regional_nodes_examined": 10, "regional_nodes_positive": 1}]
    ).to_csv(inp, index=False)
    out = tmp_path / "preds.csv"
    assert (
        main(["predict", str(inp), "--model-path", str(tmp_path / "model.joblib"), "-o", str(out)])
        == 0
    )
    preds = pd.read_csv(out)
    assert {"probability_dead", "predicted_status"} <= set(preds.columns)


def test_predict_missing_model_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train one first"):
        SurvivalModel.load(tmp_path / "nope.joblib")


def test_predict_missing_feature_raises(trained_model_path):
    model = SurvivalModel.load(trained_model_path)
    with pytest.raises(DataValidationError, match="tumor_size"):
        model.predict([{"age": 50, "regional_nodes_examined": 1, "regional_nodes_positive": 0}])


def test_threshold_controls_label(trained_model_path):
    row = [
        {"age": 50, "tumor_size": 20, "regional_nodes_examined": 10, "regional_nodes_positive": 1}
    ]
    assert (
        SurvivalModel.load(trained_model_path, threshold=0.0).predict(row)[0].predicted_status
        == "Dead"
    )
    assert (
        SurvivalModel.load(trained_model_path, threshold=1.01).predict(row)[0].predicted_status
        == "Alive"
    )
