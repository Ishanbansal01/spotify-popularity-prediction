import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import os

with open("artifacts/data.pkl", "rb") as f:
    X_train, X_test, y_train, y_test, feature_names = pickle.load(f)

os.makedirs("figures", exist_ok=True)

df = pd.DataFrame(
    np.vstack([X_train, X_test]),
    columns=feature_names
)
y_all = pd.concat([y_train, y_test]).reset_index(drop=True)
df["popular"] = y_all

print("Generating extra figures...\n")

try:
    counts = df["popular"].value_counts().sort_index()
    labels = ["Not Popular", "Popular"]
    colors = ["#90CAF9", "#4CAF50"]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, counts.values, color=colors, edgecolor="white", width=0.5)

    for bar, count in zip(bars, counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 300,
            f"{count:,}\n({count/len(df)*100:.1f}%)",
            ha="center", va="bottom", fontsize=12
        )

    plt.ylabel("Number of Tracks", fontsize=13)
    plt.title("Class Distribution: Popular vs Not Popular", fontsize=15)
    plt.ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    plt.savefig("figures/class_distribution.png", dpi=150)
    plt.close()
    print("Saved: figures/class_distribution.png")
except Exception as e:
    print(f"Class distribution error: {e}")

try:
    audio_features = [
        "danceability", "energy", "loudness",
        "acousticness", "valence", "tempo"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, feature in enumerate(audio_features):
        popular = df[df["popular"] == 1][feature]
        not_popular = df[df["popular"] == 0][feature]

        axes[i].hist(not_popular, bins=40, alpha=0.6, color="#90CAF9", label="Not Popular", density=True)
        axes[i].hist(popular, bins=40, alpha=0.6, color="#4CAF50", label="Popular", density=True)
        axes[i].set_title(feature.capitalize(), fontsize=12)
        axes[i].set_xlabel("Value", fontsize=10)
        axes[i].set_ylabel("Density", fontsize=10)
        axes[i].legend(fontsize=9)

    fig.suptitle("Audio Feature Distributions: Popular vs Not Popular", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig("figures/feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/feature_distributions.png")
except Exception as e:
    print(f"Feature distribution error: {e}")

print("\nAll extra figures saved to figures/")