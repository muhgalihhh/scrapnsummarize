import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset JSON dengan encoding utf-8
with open('data.json', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# Preprocessing: Ambil deskripsi film
descriptions = [movie['overview'] for movie in data]
titles = [movie['name'] for movie in data]

# Langkah 1: Ekstraksi Fitur menggunakan TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(descriptions)

def recommend_movie(user_input):
    # Langkah 2: Preprocessing input pengguna
    user_input_tfidf = vectorizer.transform([user_input])

    # Langkah 3: Hitung kesamaan
    cosine_similarities = cosine_similarity(user_input_tfidf, tfidf_matrix).flatten()

    # Langkah 4: Ambil film dengan skor tertinggi
    top_indices = cosine_similarities.argsort()[-5:][::-1]  # Top 5 rekomendasi
    recommendations = [(titles[i], cosine_similarities[i]) for i in top_indices]

    return recommendations

# Contoh penggunaan
user_query = "give me a movie about space battles"
recommendations = recommend_movie(user_query)

# Output rekomendasi
print("Rekomendasi film:")
for title, score in recommendations:
    print(f"- {title} (Skor kesamaan: {score:.2f})")
