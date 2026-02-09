import pandas as pd
import zipfile
from dtsc330_26 import reusable_classifier
from dtsc330_26.readers import har
import os

zip_path = r"C:/Users/alexk/Downloads/motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0.zip"
extract_parent = r"C:/Users/alexk/Downloads"

# Extract the zip if not already extracted
top_level_folder = "motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0"
extract_dir = os.path.join(extract_parent, top_level_folder)

if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_parent)
    print("Zip extracted.")

# Load HAR data
har_reader = har.HAR(path=extract_dir, n_people=3)
df = har_reader.df
print(df.head())

# Train/test setup
labels = df['is_sleep']
features = df.drop(columns=['is_sleep', 'person', 'timestamp'])

# Train and assess classifier
rc = reusable_classifier.ReusableClassifier()
print(rc.assess(features, labels))
