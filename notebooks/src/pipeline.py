import pandas as pd
import numpy as np

df = pd.read_csv("data/Musical_instruments_reviews.csv")

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

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")


top_popular = train_df['asin'].value_counts().head(10).index.tolist()

def recommend_most_popular(user_id, k=10):
    return top_popular[:k]

print("Топ-10 популярных товаров:", top_popular)