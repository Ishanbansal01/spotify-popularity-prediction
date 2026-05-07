import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    ConfusionMatrixDisplay
)
import os

with open("artifacts/data.pkl", "rb") as f:
    X_train, X_test, y_train, y_test, feature_names = pickle.load(f)

with open("artifacts/models.pkl", "rb") as f:
    results = pickle.load(f)

with open("artifacts/best_model_name.pkl", "rb") as f:
    best_name = pickle.load(f)

os.makedirs("figures", exist_ok=True)
print("Generating figures...\n")

try:
    plt.figure(figsize=(8, 6))
    colors = {"Logistic Regression": "#2196F3", "Random Forest": "#4CAF50", "XGBoost": "#FF5722"}

    for name, res in results.items():
        y_prob = res["model"].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", color=colors[name], linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Baseline (AUC = 0.500)")
    plt.xlabel("False Positive Rate", fontsize=13)
    plt.ylabel("True Positive Rate", fontsize=13)
    plt.title("ROC Curve Comparison", fontsize=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig("figures/roc_curve.png", dpi=150)
    plt.close()
    print("Saved: figures/roc_curve.png")
except Exception as e:
    print(f"ROC curve error: {e}")

try:
    rf_model = results["Random Forest"]["model"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(9, 6))
    plt.barh(top_features[::-1], top_importances[::-1], color="#4CAF50")
    plt.xlabel("Importance Score", fontsize=13)
    plt.title("Top 15 Feature Importances (Random Forest)", fontsize=15)
    plt.tight_layout()
    plt.savefig("figures/feature_importance.png", dpi=150)
    plt.close()
    print("Saved: figures/feature_importance.png")
except Exception as e:
    print(f"Feature importance error: {e}")

try:
    best_model = results[best_name]["model"]
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Popular", "Popular"])

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {best_name}", fontsize=15)
    plt.tight_layout()
    plt.savefig("figures/confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: figures/confusion_matrix.png")
except Exception as e:
    print(f"Confusion matrix error: {e}")

try:
    audio_features = [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence", "tempo"
    ]

    df_corr = pd.DataFrame(X_train, columns=feature_names)
    corr = df_corr[audio_features].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(audio_features)))
    ax.set_yticks(range(len(audio_features)))
    ax.set_xticklabels(audio_features, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(audio_features, fontsize=10)

    for i in range(len(audio_features)):
        for j in range(len(audio_features)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title("Audio Feature Correlation Heatmap", fontsize=15)
    plt.tight_layout()
    plt.savefig("figures/correlation_heatmap.png", dpi=150)
    plt.close()
    print("Saved: figures/correlation_heatmap.png")
except Exception as e:
    print(f"Correlation heatmap error: {e}")

print("\n── Final Results Summary ────────────────────────────────")
print(f"{'Model':<25} {'Accuracy':>10} {'F1 Score':>10} {'AUC-ROC':>10}")
print("-" * 57)
for name, res in results.items():
    print(f"{name:<25} {res['accuracy']:>10} {res['f1']:>10} {res['auc']:>10}")
print(f"\nBest model: {best_name}")
print("\nAll figures saved to figures/")