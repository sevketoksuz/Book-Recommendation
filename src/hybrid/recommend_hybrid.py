import pandas as pd
from baseline.recommend_popular import recommend_popular
from collaborativeFiltering.recommend_cf import recommend_cf
from contentBasedFiltering.recommend_cbf import recommend_cbf

def recommend_hybrid(user_id, ratings_df, books_df,
                     model_cf, tfidf_matrix, indices,
                     top_books_df, n=10):
    """
    Returns hybrid recommendation results for a given user.
    Chooses between CF, CBF, or Popularity depending on user activity.
    """
    
    rating_count = ratings_df[ratings_df['User-ID'] == user_id].shape[0]

    if rating_count >= 10:
        try:
            return recommend_cf(user_id, model_cf, ratings_df, books_df, n)
        except Exception as e:
            print(f"[CF FAIL] Fallback to CBF for user {user_id} — Reason: {e}")
            return recommend_cbf(user_id, ratings_df, books_df, tfidf_matrix, indices, n)
        
    elif 1 <= rating_count < 10:
        try:
            return recommend_cbf(user_id, ratings_df, books_df, tfidf_matrix, indices, n)
        except Exception as e:
            print(f"[CBF FAIL] Fallback to Popularity for user {user_id} — Reason: {e}")
            return recommend_popular(user_id, top_books_df, n)
        
    else:
        return recommend_popular(user_id, top_books_df, n)
