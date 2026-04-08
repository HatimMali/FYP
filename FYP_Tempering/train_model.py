import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import joblib

# -------- CONFIG --------
FEATURES_PATH = "features.npy"
LABELS_PATH   = "labels.npy"
MODEL_PATH    = "rf_model1.joblib"  # filename kept same so app.py works unchanged
# ------------------------


def main():
    # ---------------- LOAD ----------------
    X = np.load(FEATURES_PATH)
    y = np.load(LABELS_PATH)

    print("Loaded features:", X.shape)
    print("Loaded labels:  ", y.shape)

    # ---------------- SCALING ----------------
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ---------------- SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("\nTrain samples:", len(y_train))
    print("Test samples :", len(y_test))

    # ---------------- CLASS WEIGHT ----------------
    # XGBoost uses scale_pos_weight instead of class_weight="balanced"
    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    scale = neg / pos
    print(f"\nClass balance — Authentic: {neg}  Tampered: {pos}  scale_pos_weight: {scale:.2f}")

    # ---------------- MODEL ----------------
    clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,   # handles class imbalance
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining model...")
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50               # prints loss every 50 rounds
    )
    print("Training complete.")

    # ---------------- PREDICTION ----------------
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # ---------------- METRICS ----------------
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print("\n" + "="*50)
    print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"ROC-AUC  : {roc:.4f}")
    print("="*50)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ---------------- FEATURE IMPORTANCE ----------------
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nTop 10 Important Features:")
    print("-" * 40)
    for rank, i in enumerate(indices[:10], start=1):
        print(f"{rank:>2}. Feature {i:<3}  Importance: {importances[i]:.5f}")

    # ---------------- SAVE MODEL ----------------
    joblib.dump((scaler, clf), MODEL_PATH)
    print(f"\nModel + scaler saved to '{MODEL_PATH}'")


if __name__ == "__main__":
    main()