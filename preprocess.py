import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

df = pd.read_csv("data/dataset.csv")
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

df = df.drop(columns=["Unnamed: 0", "track_id", "artists", "album_name", "track_name", "track_genre"])

df = df.dropna()
print(f"After dropping nulls: {df.shape[0]} rows")

df = df.drop_duplicates()
print(f"After dropping duplicates: {df.shape[0]} rows")

df["popular"] = (df["popularity"] >= 50).astype(int)
df = df.drop(columns=["popularity"])
print(f"Class distribution:\n{df['popular'].value_counts()}")

df["explicit"] = df["explicit"].astype(int)

df = pd.get_dummies(df, columns=["key", "mode"], drop_first=True)
print(f"Shape after encoding: {df.shape}")

X = df.drop(columns=["popular"])
y = df["popular"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape[0]} rows")
print(f"Testing set:  {X_test.shape[0]} rows")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs("artifacts", exist_ok=True)

with open("artifacts/data.pkl", "wb") as f:
    pickle.dump((X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()), f)

with open("artifacts/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nPreprocessing complete. Artifacts saved to artifacts/")