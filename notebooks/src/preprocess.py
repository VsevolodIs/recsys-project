import pandas as pd

df = pd.read_csv("data/Musical_instruments_reviews.csv")
print(df[['reviewerID', 'asin', 'overall', 'unixReviewTime']].head())