# Spotify Track Popularity Prediction

Predicting whether a Spotify track is popular using audio features and ensemble learning.

This project investigates whether the intrinsic audio characteristics of a Spotify track
can predict its commercial popularity. We formulate this as a binary classification problem
and compare three machine learning models: Logistic Regression, Random Forest, and XGBoost.

## Dataset
The dataset I used can be downloaded on kaggle with the link below that I have attached.
https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

Place the downloaded file in a folder called "data" so the path looks like this --> `data/dataset.csv`

## File Structure
- preprocess.py —-> loads and cleans the dataset, engineers the binary label, encodes
  categorical features, scales numeric features, and saves the train/test split
- train.py —-> trains Logistic Regression, Random Forest, and XGBoost classifiers
  and saves all models and evaluation metrics
- evaluate.py —-> generates the ROC curve, feature importance chart, confusion matrix,
  and correlation heatmap
- visualize.py —-> generates the class distribution chart and audio feature
  distribution plots

## How to Run

Step 1 — Install all required libraries listed here:
pip install pandas numpy scikit-learn xgboost matplotlib

Step 2 — Run preprocessing using following command:
python3 preprocess.py

Step 3 — Train all models using following command:
python3 train.py

Step 4 — Generate evaluation figures using following command:
python3 evaluate.py

Step 5 — Generate additional figures using following command:
python3 visualize.py

## Results
When everything is ran, there will be a folder created called 'figures' which will have all my graphs, plots, and other visuals that were made with the code. 
