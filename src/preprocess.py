import os
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "Musical_instruments_reviews.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

print("Загрузка сырых данных...")
df = pd.read_csv(RAW_DATA_PATH)

df = df[['reviewerID', 'asin', 'overall', 'unixReviewTime']].dropna()

user_counts = df['reviewerID'].value_counts()
item_counts = df['asin'].value_counts()

df = df[df['reviewerID'].isin(user_counts[user_counts >= 5].index)]
df = df[df['asin'].isin(item_counts[item_counts >= 5].index)]

df = df.sort_values(by=['reviewerID', 'unixReviewTime'])


def time_split(group):
  n = len(group)
  train_n = int(np.floor(0.8 * n))
  return group.iloc[:train_n], group.iloc[train_n:]


train_list = []
test_list = []

for _, group in df.groupby('reviewerID'):
  if len(group) >= 2:
    tr, te = time_split(group)
    train_list.append(tr)
    test_list.append(te)

train_df = pd.concat(train_list).reset_index(drop=True)
test_df = pd.concat(test_list).reset_index(drop=True)

print(f"Готово! Train size: {len(train_df)}, Test size: {len(test_df)}")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
print(f"Файлы успешно сохранены в папку: {PROCESSED_DIR}")