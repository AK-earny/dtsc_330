import pandas as pd
import zipfile
from dtsc330_26 import reusable_classifier
from dtsc330_26.readers import har
import os

# zip_path = r"C:/Users/alexk/Downloads/motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0.zip"
# extract_parent = r"C:/Users/alexk/Downloads"

# # Extract the zip if not already extracted
# top_level_folder = "motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0"
# extract_dir = os.path.join(extract_parent, top_level_folder)

# if not os.path.exists(extract_dir):
#     with zipfile.ZipFile(zip_path, 'r') as zf:
#         zf.extractall(extract_parent)
#     print("Zip extracted.")

# # Load HAR data
# har_reader = har.HAR(path=extract_dir, n_people=3)
# df = har_reader.df
# print(df.head())

# # Train/test setup
# labels = df['is_sleep']
# features = df.drop(columns=['is_sleep', 'person', 'timestamp'])

# # Train and assess classifier
# rc = reusable_classifier.ReusableClassifier()
# print(rc.assess(features, labels))
data = har.HAR(
    r"C:\Users\alexk\Downloads\motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0",
    10,
)
full_df = data.df
full_df.index = pd.to_timedelta(full_df["timestamp"], unit="s")

# We need to loop through peopel so that we don't average across them
people = pd.unique(full_df["person"])
features, labels, test_features, test_labels = [], [], [], []
for person in people:
    print(f"Computing person {person+1}")
    df = full_df.loc[full_df["person"] == person]
    # Compute a whole bunch of window/column features
    for window in ["10s", "1min", "10min", "1h", "10h"]:
        for column in ["hr", "acc_x", "acc_y", "acc_z"]:
            for fn in ["mean", "min", "max", "std"]:
                df[f"{column}_{fn}_{window}"] = df[column].rolling(window).agg(fn)

    # Then downsample the data. The lowest frequency we care about is 10s
    df = df.resample("10s").first().dropna(how="any")

    # Extract features, labels, and classify
    fs = df.drop(columns=["timestamp", "hr", "acc_x", "acc_y", "acc_z", "is_sleep"])
    ls = df["is_sleep"]

    if person < 1:
        test_features.append(fs)
        test_labels.append(ls)
    else:
        features.append(fs)
        labels.append(ls)

classifier = reusable_classifier.ReusableClassifier("random_forest")
classifier.train(pd.concat(features), pd.concat(labels))

pred_labels = classifier.predict(pd.concat(test_features))
test_labels = pd.concat(test_labels)

count_equal = (pred_labels.astype(int) == test_labels.to_numpy().astype(int)).sum()
print(count_equal / len(test_labels))

classifier = reusable_classifier.ReusableClassifier("xgboost")
classifier.train(pd.concat(features), pd.concat(labels))

pred_labels = classifier.predict(pd.concat(test_features))

count_equal = (pred_labels.astype(int) == test_labels.to_numpy().astype(int)).sum()
print(count_equal / len(test_labels))
