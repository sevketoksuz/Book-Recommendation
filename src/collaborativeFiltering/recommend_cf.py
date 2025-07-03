import pandas as pd

def recommend_cf(user_id, model, ratings_df, books_df, n=10):
    all_books = books_df['ISBN'].tolist()
    rated_books = ratings_df[ratings_df['User-ID'] == user_id]['ISBN'].tolist()
    unrated_books = [isbn for isbn in all_books if isbn not in rated_books]

    predictions = [model.predict(user_id, isbn) for isbn in unrated_books]
    predictions.sort(key=lambda x: x.est, reverse=True)

    top_preds = predictions[:n]
    book_title_map = dict(zip(books_df['ISBN'], books_df['Title']))

    results = []
    for pred in top_preds:
        title = book_title_map.get(pred.iid, "Unknown Title")
        results.append({
            "User-ID": user_id,
            "Title": title,
            "Score": round(pred.est, 2),
            "Source": "CF"
        })
    return pd.DataFrame(results)
