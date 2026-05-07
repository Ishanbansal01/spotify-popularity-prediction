import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import time

with open("artifacts/data.pkl", "rb") as f:
    X_train, X_test, y_train, y_test, feature_names = pickle.load(f)

print("Data loaded successfully")
print(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features\n")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    )
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    start = time.time()
    
    model.fit(X_train, y_train)
    duration = round(time.time() - start, 2)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = round(accuracy_score(y_test, y_pred), 4)
    f1 = round(f1_score(y_test, y_pred), 4)
    auc = round(roc_auc_score(y_test, y_prob), 4)
    
    results[name] = {
        "model": model,
        "accuracy": accuracy,
        "f1": f1,
        "auc": auc
    }
    
    print(f"  Accuracy : {accuracy}")
    print(f"  F1 Score : {f1}")
    print(f"  AUC-ROC  : {auc}")
    print(f"  Time     : {duration}s\n")

best_name = max(results, key=lambda x: results[x]["auc"])
print(f"Best model by AUC-ROC: {best_name}")

with open("artifacts/models.pkl", "wb") as f:
    pickle.dump(results, f)

with open("artifacts/best_model_name.pkl", "wb") as f:
    pickle.dump(best_name, f)

print("\nAll models saved to artifacts/")