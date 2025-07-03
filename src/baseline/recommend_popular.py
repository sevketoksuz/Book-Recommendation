import pandas as pd

def recommend_popular(user_id, top_books_df, n=10):
    recs = top_books_df.head(n).copy()
    recs["User-ID"] = user_id
    recs["Score"] = recs["RatingCount"]
    recs["Source"] = "POPULARITY"
    return recs[["User-ID", "Title", "Score", "Source"]]
