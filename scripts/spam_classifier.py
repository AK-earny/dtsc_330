"""
Stopping Spam using: https://www.kaggle.com/datasets/ssssws/spam-email-detection-dataset-clean-and-ml-ready

"""
import pandas as pd
import zipfile
from dtsc330_26 import reusable_classifier
from sklearn.model_selection import train_test_split ## added with ai assistence


data = "C:/Users/alexk/Downloads/archive.zip"

import zipfile

with zipfile.ZipFile(data, 'r') as zip_ref:
    zip_ref.extractall("spam_data")

df = pd.read_csv("spam_data/spam_email_dataset.csv")

df = df.drop(columns=["email_id", "subject", "email_text", "sender_email"])

df = pd.get_dummies(df, columns=["sender_domain", "email_day_of_week"], drop_first=True)

features = df.drop(columns=["label"])
labels = df["label"]


features, test_features, labels, test_labels = train_test_split(features, labels) ## added with ai assistence

classifier = reusable_classifier.ReusableClassifier("xgboost")

classifier.train(features, labels)

pred_labels = classifier.predict(test_features)

count_equal = (pred_labels.astype(int) == test_labels.to_numpy().astype(int)).sum()
print(count_equal / len(test_labels))
"""
When run with XGBoost the model seems to be 100% accurate
When run with a less powerful model like a random forest, seems to be close to 100"
"""