from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

training_data = []
vectorizer = None
matrix = None

def add_example(example):
    training_data.append(example)

def build_index():
    global vectorizer, matrix
    corpus = [ex.email for ex in training_data]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)

def find_best_match(query: str):
    if not matrix or not vectorizer:
        return None, 0.0
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, matrix).flatten()
    best_idx = np.argmax(sims)
    return training_data[best_idx], sims[best_idx]
