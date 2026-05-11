import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os

with open("artifacts/data.pkl", "rb") as f:
    X_train, X_test, y_train, y_test, feature_names = pickle.load(f)

os.makedirs("figures", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

X_all = np.vstack([X_train, X_test])
y_all = pd.concat([y_train, y_test]).reset_index(drop=True).values

print("Finding optimal number of clusters...\n")

inertias = []
silhouette_scores = []
k_range = range(2, 9)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_all)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_all, km.labels_, sample_size=5000, random_state=42))
    print(f"  k={k} | Inertia: {km.inertia_:.0f} | Silhouette: {silhouette_scores[-1]:.4f}")

best_k = k_range[np.argmax(silhouette_scores)]
print(f"\nBest k by silhouette score: {best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_all)

with open("artifacts/kmeans.pkl", "wb") as f:
    pickle.dump(kmeans, f)

print(f"\nCluster distribution:")
unique, counts = np.unique(cluster_labels, return_counts=True)
for c, n in zip(unique, counts):
    print(f"  Cluster {c}: {n} tracks")

print("\nGenerating cluster figures...\n")

try:
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_all)

    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800", "#00BCD4", "#E91E63", "#795548"]
    plt.figure(figsize=(10, 7))

    for i in range(best_k):
        mask = cluster_labels == i
        plt.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=colors[i], label=f"Cluster {i}",
            alpha=0.4, s=8
        )

    plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=12)
    plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=12)
    plt.title("K-Means Cluster Visualization (PCA Projection)", fontsize=15)
    plt.legend(fontsize=10, markerscale=3)
    plt.tight_layout()
    plt.savefig("figures/cluster_pca.png", dpi=150)
    plt.close()
    print("Saved: figures/cluster_pca.png")
except Exception as e:
    print(f"Cluster PCA error: {e}")

try:
    popularity_rates = []
    cluster_sizes = []

    for i in range(best_k):
        mask = cluster_labels == i
        rate = y_all[mask].mean() * 100
        popularity_rates.append(rate)
        cluster_sizes.append(mask.sum())
        print(f"  Cluster {i}: {rate:.1f}% popular ({mask.sum()} tracks)")

    cluster_names = [f"Cluster {i}" for i in range(best_k)]
    bar_colors = [colors[i] for i in range(best_k)]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(cluster_names, popularity_rates, color=bar_colors, edgecolor="white", width=0.6)

    for bar, rate, size in zip(bars, popularity_rates, cluster_sizes):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{rate:.1f}%\n(n={size:,})",
            ha="center", va="bottom", fontsize=10
        )

    plt.axhline(y=y_all.mean()*100, color="black", linestyle="--", linewidth=1.5,
                label=f"Overall average ({y_all.mean()*100:.1f}%)")
    plt.ylabel("Popularity Rate (%)", fontsize=13)
    plt.title("Popularity Rate by K-Means Cluster", fontsize=15)
    plt.legend(fontsize=11)
    plt.ylim(0, max(popularity_rates) * 1.2)
    plt.tight_layout()
    plt.savefig("figures/cluster_popularity.png", dpi=150)
    plt.close()
    print("Saved: figures/cluster_popularity.png")
except Exception as e:
    print(f"Cluster popularity error: {e}")

print("\nClustering complete.")