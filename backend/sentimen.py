import re

import nltk
import torch
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from torch.nn.functional import softmax
from transformers import BertForSequenceClassification, BertTokenizer

# Download stopwords jika belum ada
nltk.download('stopwords')
nltk.download('punkt')

# Load stopwords untuk bahasa Indonesia dan Inggris
stopwords_indonesia = stopwords.words('indonesian')
stopwords_english = stopwords.words('english')
stopwords_combined = stopwords_indonesia + stopwords_english

# Fungsi untuk preprocess teks
def preprocess_text(berita):
    # Tokenisasi dan penghapusan stopwords
    tokens = word_tokenize(berita.lower())  # Tokenisasi kata
    tokens_cleaned = [word for word in tokens if word not in stopwords_combined and re.match(r'\w+', word)]
    return " ".join(tokens_cleaned)

# Fungsi analisis sentimen menggunakan VADER
def vader_sentiment_analysis(berita):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(berita)
    sentiment = 'Netral'
    if sentiment_scores['compound'] >= 0.05:
        sentiment = 'Positif'
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = 'Negatif'
    return sentiment, sentiment_scores

# Fungsi analisis sentimen menggunakan IndoBERT
def indobert_sentiment_analysis(berita):
    # Load IndoBERT model dan tokenizer
    model_name = "indobert-base-uncased"  # Model IndoBERT
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    
    # Tokenisasi teks
    inputs = tokenizer(berita, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Prediksi sentimen
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Konversi logits ke probabilitas menggunakan softmax
    probabilities = softmax(logits, dim=-1)
    
    # Ambil label sentimen dengan probabilitas tertinggi
    sentiment = torch.argmax(probabilities, dim=-1).item()
    
    # Tentukan sentimen berdasarkan model IndoBERT
    if sentiment == 0:
        return 'Negatif', probabilities
    elif sentiment == 1:
        return 'Positif', probabilities
    else:
        return 'Netral', probabilities

# Berita campuran (Indo + English)
berita = "Pemerintah baru saja mengumumkan kebijakan yang sangat positif bagi perekonomian. The economy is improving."

# Preprocessing teks
berita_cleaned = preprocess_text(berita)

# Analisis Sentimen menggunakan VADER
vader_sentiment, vader_scores = vader_sentiment_analysis(berita_cleaned)
print("Sentimen VADER:", vader_sentiment)
print("VADER Scores:", vader_scores)

# Analisis Sentimen menggunakan IndoBERT
indobert_sentiment, indobert_probabilities = indobert_sentiment_analysis(berita)
print("Sentimen IndoBERT:", indobert_sentiment)
print("IndoBERT Probabilities:", indobert_probabilities)
