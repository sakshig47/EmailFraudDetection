#!/usr/bin/env python3
# =============================================================================
# main.py  —  Master orchestration script
# Run the full pipeline: download → preprocess → train → evaluate → save
# =============================================================================
# Usage:
#   python main.py                    # full pipeline
#   python main.py --skip-download    # use existing data/raw/
#   python main.py --predict "text"   # classify a single email
# =============================================================================

import argparse
import sys

from src.config import PROCESSED_DATA_FILE
from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_dataframe, save_processed
from src.models.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    print_classification_report,
    print_metrics,
)
from src.models.predict import predict_single
from src.models.train import run_training
from src.utils.helpers import dataset_summary, timer


@timer
def run_full_pipeline():
    """
    End-to-end pipeline:
    1. Load raw data
    2. EDA summary
    3. Preprocess text
    4. Train & tune all models
    5. Evaluate best model on test set
    6. Save artifacts
    """
    import pandas as pd

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 1: Loading Dataset")
    print("="*60)
    df = load_raw_data()
    dataset_summary(df, label_col="label")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 2: Preprocessing")
    print("="*60)
    df = preprocess_dataframe(df)
    save_processed(df)
    print("Sample cleaned text:")
    print(df[["message", "clean_text", "label_encoded"]].head(3).to_string(index=False))

    # ── 3. Train ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 3: Training & Hyperparameter Tuning")
    print("="*60)
    best_pipeline, X_test, y_test = run_training(df)

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 4: Final Evaluation on Test Set")
    print("="*60)
    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)
    print_metrics(metrics, model_name="Best Pipeline")
    print_classification_report(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred, model_name="Best Pipeline",
                          save_path="confusion_matrix.png")

    print("\n" + "="*60)
    print("  Pipeline complete!  Artifacts saved to models/")
    print("  Run the app:  streamlit run app/app.py")
    print("="*60)


def run_prediction(text: str):
    """Classify a single email from the command line."""
    result = predict_single(text)
    print(f"\n{'='*50}")
    print(f"  Input   : {text[:80]}{'…' if len(text)>80 else ''}")
    print(f"  Label   : {result['label']}")
    print(f"  Spam %  : {result['spam_prob']*100:.1f}%")
    print(f"  Conf    : {result['confidence']*100:.1f}%")
    print(f"{'='*50}")


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Email Fraud Detection Pipeline")
    parser.add_argument(
        "--predict", type=str, default=None,
        help="Classify a single email text string",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip dataset download (use existing data/raw/)",
    )
    args = parser.parse_args()

    if args.predict:
        run_prediction(args.predict)
    else:
        run_full_pipeline()
