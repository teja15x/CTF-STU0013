#!/usr/bin/env python3
"""
solver.py

Usage:
  python solver.py --books books.csv --reviews reviews.csv --id STU160

Outputs:
  - flags.txt (FLAG1, FLAG2, FLAG3)
  - prints progress and intermediate results
"""
import argparse
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

SUPERLATIVES = {
    "best", "perfect", "amazing", "excellent", "flawless", "outstanding",
    "unbelievable", "incredible", "fantastic", "superb", "brilliant", "wonderful"
}
DOMAIN_WORDS = {
    "character", "plot", "story", "development", "writing", "author",
    "chapter", "dialogue", "prose", "narrative", "theme"
}


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def first_n_non_space(s: str, n: int) -> str:
    res = []
    for ch in s:
        if ch.isspace():
            continue
        res.append(ch)
        if len(res) >= n:
            break
    return ''.join(res)


def guess_column(df_cols, candidates):
    for c in candidates:
        if c in df_cols:
            return c
    # case-insensitive match
    lower = {c.lower(): c for c in df_cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def prepare_reviews_df(df):
    # normalize some columns
    cols = df.columns.tolist()
    text_col = guess_column(cols, ['text', 'review', 'review_text', 'content'])
    rating_col = guess_column(cols, ['rating', 'ratings', 'stars', 'rating_value', 'score'])
    bookid_col = guess_column(cols, ['book_id', 'bookId', 'asin', 'id', 'bookid'])
    if text_col is None:
        raise ValueError(f"Could not find a text/review column in reviews CSV. Columns: {cols}")
    return df, text_col, rating_col, bookid_col


def prepare_books_df(df):
    cols = df.columns.tolist()
    title_col = guess_column(cols, ['title', 'book_title', 'name'])
    rating_num_col = guess_column(cols, ['rating_number', 'ratings_count', 'ratings', 'rating_count', 'ratingsnumber'])
    avg_col = guess_column(cols, ['average_rating', 'averageRating', 'average_rating_value', 'avg_rating'])
    bookid_col = guess_column(cols, ['book_id', 'bookId', 'id', 'asin'])
    return df, title_col, rating_num_col, avg_col, bookid_col


def count_superlatives(text):
    if not isinstance(text, str):
        return 0
    tokens = re.findall(r'\w+', text.lower())
    return sum(1 for t in tokens if t in SUPERLATIVES)


def count_domain_words(text):
    if not isinstance(text, str):
        return 0
    tokens = re.findall(r'\w+', text.lower())
    return sum(1 for t in tokens if t in DOMAIN_WORDS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--books', required=True, help='Path to books CSV')
    parser.add_argument('--reviews', required=True, help='Path to reviews CSV')
    parser.add_argument('--id', default='STU160', help='Your ID string (default: STU160)')
    parser.add_argument('--min_genuine_prob', type=float, default=0.30, help='Threshold for genuine probability (lower = more genuine)')
    args = parser.parse_args()

    id_str = args.id
    seed_hash = sha256_hex(id_str)[:8].upper()
    print(f"[+] ID string: {id_str} -> seed hash (first8 uppercase): {seed_hash}")

    # Load CSVs
    books = pd.read_csv(args.books, dtype=str, low_memory=False)
    reviews = pd.read_csv(args.reviews, dtype=str, low_memory=False)

    # prepare dataframes and detect columns
    books, title_col, rating_num_col, avg_col, books_id_col = prepare_books_df(books)
    reviews, review_text_col, review_rating_col, reviews_bookid_col = prepare_reviews_df(reviews)

    print(f"[+] Detected books title column: {title_col}; rating_number col: {rating_num_col}; average col: {avg_col}; book id col: {books_id_col}")
    print(f"[+] Detected reviews text column: {review_text_col}; rating col: {review_rating_col}; book id col: {reviews_bookid_col}")

    # Convert numeric columns safely
    def to_numeric_column(df, col):
        if col is None:
            return None
        try:
            return pd.to_numeric(df[col], errors='coerce')
        except Exception:
            return None

    books_rating_num = to_numeric_column(books, rating_num_col)
    books_avg = to_numeric_column(books, avg_col)

    # Filter books with rating_number == 1234 and average_rating == 5.0
    candidate_mask = pd.Series(False, index=books.index)
    if rating_num_col and avg_col:
        candidate_mask = (books_rating_num == 1234) & (np.isclose(books_avg.fillna(-1).astype(float), 5.0))
    else:
        print("[!] Could not find rating_number and/or average_rating columns; scanning titles instead.")

    candidates = books[candidate_mask]
    print(f"[+] Found {len(candidates)} candidate book(s) with rating_number==1234 and average_rating==5.0")

    # If none found, still continue scanning all books (rare)
    if len(candidates) == 0:
        candidates = books.copy()
        print("[*] No exact candidates found; scanning reviews for seed hash across all books.")

    # find review that contains the seed hash (case-insensitive) in review text
    matched_review = None
    matched_book_id = None
    matched_book_row = None

    # prepare reviews text lower for search
    reviews['_search_text'] = reviews[review_text_col].fillna('').astype(str)

    # loop through candidate book ids if available
    candidate_book_ids = None
    if books_id_col is not None:
        candidate_book_ids = set(candidates[books_id_col].astype(str).tolist())

        # filter reviews to those for candidate books if reviews have book ids
        if reviews_bookid_col is not None:
            rev_candidates = reviews[reviews[reviews_bookid_col].astype(str).isin(candidate_book_ids)]
        else:
            rev_candidates = reviews
    else:
        rev_candidates = reviews

    # search for seed hash
    pattern = re.compile(re.escape(seed_hash), re.IGNORECASE)
    found = rev_candidates[rev_candidates['_search_text'].str.contains(pattern)]
    if len(found) == 0:
        # broaden search to entire reviews if not found
        found = reviews[reviews['_search_text'].str.contains(pattern)]
    if len(found) == 0:
        print("[!] No reviews contained the seed hash. Exiting.")
        sys.exit(1)

    # pick the first match
    matched_review = found.iloc[0]
    print(f"[+] Found review containing seed hash in reviews CSV (index {found.index[0]})")
    if reviews_bookid_col is not None:
        matched_book_id = str(matched_review.get(reviews_bookid_col, '')).strip()
        print(f"[+] Matched book id from review: {matched_book_id}")
        if books_id_col is not None:
            matched_book_rows = books[books[books_id_col].astype(str) == matched_book_id]
            if len(matched_book_rows) > 0:
                matched_book_row = matched_book_rows.iloc[0]
    else:
        # try to infer book by title mention or by candidate list
        matched_book_row = None

    # If matched_book_row not found, try to find which book has reviews containing seed hash by grouping reviews
    if matched_book_row is None:
        # find book ids for all reviews that contained seed hash
        if reviews_bookid_col is not None:
            ids = found[reviews_bookid_col].dropna().astype(str).unique()
            if len(ids) > 0 and books_id_col is not None:
                matched_book_rows = books[books[books_id_col].astype(str).isin(ids)]
                if len(matched_book_rows) > 0:
                    matched_book_row = matched_book_rows.iloc[0]
                    matched_book_id = str(matched_book_row.get(books_id_col, '')).strip()

    if matched_book_row is None:
        # As a last resort, if candidates set has only one book, use that
        if len(candidates) == 1:
            matched_book_row = candidates.iloc[0]
            matched_book_id = str(matched_book_row.get(books_id_col, '')).strip() if books_id_col else ''
            print("[*] Using sole candidate book as match.")
        else:
            print("[!] Unable to identify the book row automatically. Please provide sample rows or column names.")
            sys.exit(1)

    title = str(matched_book_row.get(title_col, '')).strip() if title_col else ''
    print(f"[+] Identified book title: {title}")

    # FLAG2 is simply the seed hash uppercase in braces
    FLAG2_value = seed_hash.upper()
    FLAG2 = f"FLAG2{{{FLAG2_value}}}"
    print(f"[+] FLAG2 = {FLAG2}")

    # FLAG1: take first 8 non-space characters of the book's title then SHA256
    first8chars = first_n_non_space(title, 8)
    FLAG1_hash = sha256_hex(first8chars)
    FLAG1 = FLAG1_hash
    print(f"[+] First 8 non-space chars of title: '{first8chars}' -> SHA256 -> FLAG1 = {FLAG1}")

    # Now Step 3: Train a model and use SHAP to find top-3 words that reduce suspicion.
    # Collect reviews for the matched book
    if reviews_bookid_col is not None and matched_book_id:
        book_reviews = reviews[reviews[reviews_bookid_col].astype(str) == matched_book_id].copy()
    else:
        # If no book id linking, attempt to find reviews that mention the title
        book_reviews = reviews[reviews['_search_text'].str.contains(re.escape(title), case=False, na=False)].copy()
        if len(book_reviews) == 0:
            # fallback: use all reviews that are for the candidate book(s)
            book_reviews = reviews.copy()

    print(f"[+] Found {len(book_reviews)} reviews for the target book (or fallback set).")

    # Add derived features
    book_reviews['text_clean'] = book_reviews[review_text_col].fillna('').astype(str)
    book_reviews['word_count'] = book_reviews['text_clean'].apply(lambda t: len(re.findall(r'\w+', t)))
    book_reviews['char_count'] = book_reviews['text_clean'].apply(lambda t: len(t))
    book_reviews['super_count'] = book_reviews['text_clean'].apply(count_superlatives)
    book_reviews['domain_count'] = book_reviews['text_clean'].apply(count_domain_words)
    # normalize rating
    if review_rating_col:
        try:
            book_reviews['rating_num'] = pd.to_numeric(book_reviews[review_rating_col], errors='coerce')
        except Exception:
            book_reviews['rating_num'] = np.nan
    else:
        book_reviews['rating_num'] = np.nan

    # Heuristic labels for training:
    # suspicious = 5-star + short (<30 words) + superlatives > 0
    # genuine = 5-star + long (>60 words) + domain-specific words > 0
    suspicious_mask = (book_reviews['rating_num'] == 5) & (book_reviews['word_count'] < 30) & (book_reviews['super_count'] > 0)
    genuine_mask = (book_reviews['rating_num'] == 5) & (book_reviews['word_count'] > 60) & (book_reviews['domain_count'] > 0)

    labeled = book_reviews[suspicious_mask | genuine_mask].copy()
    labeled['label'] = np.where(suspicious_mask.loc[labeled.index], 1, 0)  # 1 = suspicious
    print(f"[+] For model training: {labeled['label'].sum()} suspicious and {(labeled.shape[0] - labeled['label'].sum())} genuine examples (total {len(labeled)}).")

    # If not enough labeled data, expand to use all reviews (global heuristic)
    if len(labeled) < 10:
        print("[*] Insufficient labeled samples for the target book; expanding labeling to all reviews for training.")
        all_reviews = reviews.copy()
        all_reviews['text_clean'] = all_reviews[review_text_col].fillna('').astype(str)
        all_reviews['word_count'] = all_reviews['text_clean'].apply(lambda t: len(re.findall(r'\w+', t)))
        all_reviews['super_count'] = all_reviews['text_clean'].apply(count_superlatives)
        all_reviews['domain_count'] = all_reviews['text_clean'].apply(count_domain_words)
        if review_rating_col:
            all_reviews['rating_num'] = pd.to_numeric(all_reviews.get(review_rating_col, pd.Series()), errors='coerce')
        else:
            all_reviews['rating_num'] = np.nan

        suspicious_mask_all = (all_reviews['rating_num'] == 5) & (all_reviews['word_count'] < 30) & (all_reviews['super_count'] > 0)
        genuine_mask_all = (all_reviews['rating_num'] == 5) & (all_reviews['word_count'] > 60) & (all_reviews['domain_count'] > 0)
        labeled = all_reviews[suspicious_mask_all | genuine_mask_all].copy()
        labeled['label'] = np.where(suspicious_mask_all.loc[labeled.index], 1, 0)
        print(f"[+] After expansion: {labeled['label'].sum()} suspicious and {(labeled.shape[0] - labeled['label'].sum())} genuine examples (total {len(labeled)}).")

    if len(labeled) < 6:
        print("[!] Still not enough labeled examples to train a model reliably. Aborting FLAG3 generation.")
        FLAG3 = "FLAG3{insufficient_data}"
    else:
        # Vectorize
        vect = TfidfVectorizer(min_df=2, max_features=5000, stop_words='english')
        X = vect.fit_transform(labeled['text_clean'])
        y = labeled['label'].astype(int).values

        # Train classifier
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
        clf.fit(X, y)
        print("[+] Trained logistic regression for suspicion detection.")

        # Prepare book reviews X
        X_book = vect.transform(book_reviews['text_clean'])
        # suspicion probability
        probs = clf.predict_proba(X_book)[:, 1]
        book_reviews = book_reviews.reset_index(drop=True)
        book_reviews['suspicion_prob'] = probs
        # select genuine reviews as those with low suspicion scores
        genuine_reviews = book_reviews[book_reviews['suspicion_prob'] <= args.min_genuine_prob]
        # fallback to bottom 30% if threshold yields too few
        if len(genuine_reviews) < max(3, int(0.3 * max(1, len(book_reviews)))):
            cutoff = np.percentile(book_reviews['suspicion_prob'], 30)
            genuine_reviews = book_reviews[book_reviews['suspicion_prob'] <= cutoff]

        print(f"[+] Selected {len(genuine_reviews)} reviews as 'genuine' for SHAP analysis (threshold {args.min_genuine_prob}).")

        # Compute SHAP values for genuine reviews
        feature_names = vect.get_feature_names_out()
        try:
            if SHAP_AVAILABLE:
                # Use LinearExplainer for linear model with transformed data
                background_idx = np.random.choice(X.shape[0], size=min(50, X.shape[0]), replace=False)
                X_background = X[background_idx].toarray()
                explainer = shap.LinearExplainer(clf, X_background, feature_perturbation="interventional")
                # compute shap values for genuine reviews
                X_genuine = vect.transform(genuine_reviews['text_clean']).toarray()
                shap_values = explainer.shap_values(X_genuine)
                # shap_values is list or array; with LogisticRegression it may return one array per class or single 2D
                if isinstance(shap_values, list):
                    # shap_values[1] corresponds to positive class contributions
                    sv = shap_values[1]
                else:
                    sv = shap_values
                mean_shap = np.mean(sv, axis=0)  # mean contribution across genuine reviews
            else:
                raise Exception("SHAP not available, using linear coefficient approximation.")
        except Exception as e:
            # Fallback: contributions = mean(X_genuine * coef)
            print(f"[*] SHAP analysis unavailable or failed ({e}). Using linear contribution approximation.")
            coef = clf.coef_.ravel()  # shape (n_features,)
            X_genuine = vect.transform(genuine_reviews['text_clean']).toarray()
            contributions = X_genuine * coef[np.newaxis, :]
            mean_shap = np.mean(contributions, axis=0)

        # We want words that reduce suspicion => most negative mean contribution
        # get indices of features sorted by mean_shap ascending
        idx_sorted = np.argsort(mean_shap)  # ascending
        top3_idx = idx_sorted[:3]
        top3_words = [feature_names[i] for i in top3_idx]
        print(f"[+] Top 3 words that reduce suspicion (in order): {top3_words}")

        # FLAG3: concatenate words + numeric ID (digits from id_str)
        digits = ''.join(re.findall(r'\d+', id_str))
        if digits == '':
            digits = '1'
        concat = ''.join(top3_words) + digits
        flag3_hash = sha256_hex(concat)[:10]
        FLAG3 = f"FLAG3{{{flag3_hash}}}"
        print(f"[+] Concatenated string: '{concat}' -> SHA256 -> first10 -> {flag3_hash} -> FLAG3 = {FLAG3}")

    # Write flags.txt
    flags_content = []
    flags_content.append(f"FLAG1 = {FLAG1}")
    flags_content.append(f"FLAG2 = {FLAG2}")
    flags_content.append(f"FLAG3 = {FLAG3}")
    with open('flags.txt', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(flags_content))
    print("[+] Wrote flags to flags.txt")
    print('\n'.join(flags_content))
    print("[+] Done.")


if __name__ == '__main__':
    main()