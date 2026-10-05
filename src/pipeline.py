from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

train_path = PROCESSED_DIR / "train.csv"
test_path = PROCESSED_DIR / "test.csv"

if not train_path.exists() or not test_path.exists():
  raise FileNotFoundError(
      "Сначала запустите preprocess.py для генерации train.csv и test.csv!"
  )

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(
    f"Данные загружены. Обучающая выборка: {len(train_df)}, Тестовая выборка:"
    f" {len(test_df)}"
)

top_popular = train_df['asin'].value_counts().head(10).index.tolist()


def recommend_most_popular(user_id=None, k=10):
  return top_popular[:k]


print("\nТоп-10 популярных товаров (Most Popular):")
print(top_popular)


def evaluate_recall_at_k(recommend_fn, test_data, k=10):
  actual_interactions = test_data.groupby('reviewerID')['asin'].apply(set)

  recalls = []
  for user_id, true_items in actual_interactions.items():
    if not true_items:
      continue
    recs = set(recommend_fn(user_id, k=k))
    hits = len(recs & true_items)
    recalls.append(hits / len(true_items))

  return np.mean(recalls)


recall_10 = evaluate_recall_at_k(recommend_most_popular, test_df, k=10)
print(f"\nБейзлайн Most Popular -> Recall@10: {recall_10:.4f}")