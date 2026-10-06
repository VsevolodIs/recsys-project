from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import implicit

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DATA_PATH = BASE_DIR / "data" / "Musical_instruments_reviews.csv"

train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

print(f"Данные загружены: Train={len(train_df)}, Test={len(test_df)}")

actual_interactions = test_df.groupby('reviewerID')['asin'].apply(set).to_dict()

top_popular = train_df['asin'].value_counts().head(10).index.tolist()


def recommend_most_popular(user_id=None, k=10):
    return top_popular[:k]


print("\nОбучение Implicit ALS...")
unique_users = train_df['reviewerID'].unique()
unique_items = train_df['asin'].unique()

user2idx = {u: i for i, u in enumerate(unique_users)}
item2idx = {it: i for i, it in enumerate(unique_items)}
idx2item = {i: it for it, i in item2idx.items()}

train_df['user_idx'] = train_df['reviewerID'].map(user2idx)
train_df['item_idx'] = train_df['asin'].map(item2idx)

rows = train_df['user_idx'].values
cols = train_df['item_idx'].values
confidence = train_df['overall'].values

user_item_matrix = csr_matrix((confidence, (rows, cols)), shape=(len(unique_users), len(unique_items)))

als_model = implicit.als.AlternatingLeastSquares(
    factors=32,
    regularization=0.05,
    iterations=20,
    random_state=42
)
als_model.fit(user_item_matrix)

def recommend_als(user_id, k=10):
    if user_id not in user2idx:
        return top_popular[:k]
    u_idx = user2idx[user_id]
    ids, scores = als_model.recommend(
        u_idx,
        user_item_matrix[u_idx],
        N=k,
        filter_already_liked_items=True
    )
    return [idx2item[i] for i in ids]


print("Обучение Content-Based (TF-IDF)...")
raw_df = pd.read_csv(RAW_DATA_PATH, usecols=['asin', 'reviewText', 'summary']).fillna('')
raw_df['text'] = raw_df['summary'] + " " + raw_df['reviewText']

item_texts = (
    raw_df[raw_df['asin'].isin(unique_items)]
    .groupby('asin')['text']
    .apply(lambda x: ' '.join(x))
    .reset_index()
)

tfidf = TfidfVectorizer(max_features=2500, stop_words='english')
tfidf_matrix = tfidf.fit_transform(item_texts['text'])
item_text_idx = {it: idx for idx, it in enumerate(item_texts['asin'])}
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend_content_based(user_id, k=10):
    user_history = train_df[train_df['reviewerID'] == user_id]['asin'].tolist()
    if not user_history:
        return top_popular[:k]

    sim_scores = np.zeros(len(item_texts))
    for item in user_history:
        if item in item_text_idx:
            idx = item_text_idx[item]
            sim_scores += cosine_sim[idx]

    for item in user_history:
        if item in item_text_idx:
            sim_scores[item_text_idx[item]] = -1

    top_indices = sim_scores.argsort()[::-1][:k]
    return item_texts.iloc[top_indices]['asin'].tolist()


def evaluate_model(rec_fn, name='Model', k=10):
    recalls = []
    ndcgs = []

    for user_id, true_items in actual_interactions.items():
        if not true_items:
            continue
        recs = rec_fn(user_id, k=k)

        hits = len(set(recs) & true_items)
        recalls.append(hits / len(true_items))

        dcg = 0.0
        for rank, item in enumerate(recs):
            if item in true_items:
                dcg += 1.0 / np.log2(rank + 2)

        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(true_items), k)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    print(f"\n--- {name} (K={k}) ---")
    print(f"Recall@{k}: {np.mean(recalls):.4f}")
    print(f"NDCG@{k}:   {np.mean(ndcgs):.4f}")

print("\nРасчет метрик качества")
evaluate_model(recommend_most_popular, name="Most Popular (Baseline)")
evaluate_model(recommend_content_based, name="Content-Based (TF-IDF)")
evaluate_model(recommend_als, name="Collaborative Filtering (ALS)")