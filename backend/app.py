from typing import Dict, List

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Load dataset
DATA_PATH = "data.json"  # Ganti dengan path dataset Anda
df = pd.read_json(DATA_PATH, lines=True)

# Load NLP models
classification_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Define FastAPI app
app = FastAPI()

# Pydantic model for request body
class Query(BaseModel):
    text: str

# Helper function to filter movies based on NER results
def filter_movies(query_results, dataset):
    """
    Filter movies based on extracted entities and intent.
    """
    filtered_movies = dataset.copy()

    # Filter by genre
    genres = [entity["word"] for entity in query_results if entity["entity"] == "GENRE"]
    if genres:
        filtered_movies = filtered_movies[
            filtered_movies["genres"].apply(lambda x: any(g["name"].lower() in genres for g in x))
        ]
    
    # Filter by theme (if exists in dataset description)
    themes = [entity["word"] for entity in query_results if entity["entity"] == "THEME"]
    if themes:
        filtered_movies = filtered_movies[
            filtered_movies["overview"].str.contains("|".join(themes), case=False, na=False)
        ]

    return filtered_movies.sort_values(by="vote_average", ascending=False).head(5)

@app.post("/recommend/")
def recommend_movies(query: Query):
    """
    Recommend movies based on user query.
    """
    try:
        # Step 1: Extract intent and entities from user query
        user_query = query.text
        classification_result = classification_model(
            user_query,
            candidate_labels=["actor", "genre", "theme", "rating", "release date"]
        )

        # Step 2: Extract named entities (e.g., genre, theme)
        ner_results = ner_model(user_query)

        # Step 3: Filter dataset based on entities
        recommended_movies = filter_movies(ner_results, df)

        # Step 4: Return recommendations or a message if no results
        if recommended_movies.empty:
            return {"message": "No recommendations found based on your query."}

        return recommended_movies[["name", "genres", "overview", "vote_average"]].to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}

