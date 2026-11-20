"""Command-line interface: `survival-predictor train | predict`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict

import pandas as pd

from .config import load_config
from .predict import SurvivalModel
from .train import train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="survival-predictor", description="Cancer patient survival prediction"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="tune, select and evaluate models; save the best")
    p_train.add_argument("--config", default="configs/default.yaml")
    p_train.add_argument("--data-path", help="override data_path from the config")
    p_train.add_argument("--model-path", help="override model_path from the config")
    p_train.add_argument(
        "--include-survival-months",
        action="store_true",
        default=None,
        help="use the leaky survival_months feature (for comparison only)",
    )

    p_pred = sub.add_parser("predict", help="score patients from a CSV file")
    p_pred.add_argument("input", help="CSV with the model's feature columns")
    p_pred.add_argument("--model-path", default="models/model.joblib")
    p_pred.add_argument("--threshold", type=float, default=0.5)
    p_pred.add_argument("-o", "--output", help="write CSV here instead of stdout")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "train":
        config = load_config(
            args.config,
            data_path=args.data_path,
            model_path=args.model_path,
            include_survival_months=args.include_survival_months,
        )
        result = train(config)
        summary = asdict(result) | {"model_path": str(result.model_path)}
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "predict":
        model = SurvivalModel.load(args.model_path, threshold=args.threshold)
        df = pd.read_csv(args.input)
        preds = model.predict(df.to_dict(orient="records"))
        out = df.assign(
            probability_dead=[p.probability_dead for p in preds],
            predicted_status=[p.predicted_status for p in preds],
        )
        if args.output:
            out.to_csv(args.output, index=False)
        else:
            out.to_csv(sys.stdout, index=False)
        return 0

    return 1  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
