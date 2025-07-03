import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

def recommend_cbf(user_id, ratings_df, books_df, tfidf_matrix, indices, n=10):
    liked_books = ratings_df[
        (ratings_df['User-ID'] == user_id) & (ratings_df['Rating'] >= 7)
    ]['ISBN'].tolist()

    liked_indices = [indices.get(isbn) for isbn in liked_books if isbn in indices]
    liked_indices = [i for i in liked_indices if i is not None]
    if not liked_indices:
        return pd.DataFrame()

    user_vector = tfidf_matrix[liked_indices].mean(axis=0)
    user_vector = np.asarray(user_vector)

    cosine_scores = linear_kernel(user_vector, tfidf_matrix).flatten()
    books_df["similarity_score"] = cosine_scores

    seen_isbns = ratings_df[ratings_df['User-ID'] == user_id]['ISBN'].tolist()
    recs = books_df[~books_df['ISBN'].isin(seen_isbns)]
    recs = recs.sort_values(by="similarity_score", ascending=False).head(n)

    recs = recs.copy()
    recs["User-ID"] = user_id
    recs["Score"] = recs["similarity_score"]
    recs["Source"] = "CBF"

    return recs[["User-ID", "Title", "Score", "Source"]]
