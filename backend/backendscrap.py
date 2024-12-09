import math
import os
import re

import nltk
import pandas as pd
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

app = Flask(__name__)
CORS(app)

class BeritaScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.selector_template = {
            'detik.com': {
                'judul': '.detail__title',
                'penulis': '.detail__author',
                'tanggal': '.detail__date',
                'isi': '.detail__body-text > p:not(.para_caption):not(.ads):not([class]):not([id])'
            },
            'kompas.com': {
                'judul': '.read__title',
                'penulis': '.credit-title-name',
                'tanggal': '.read__time',
                'isi': '.read__content p:not(.read__more):not(.read__share):not([class]):not([id]):not(strong)'
            },
            'cnnindonesia.com': {
                'judul': 'h1.leading-9',
                'penulis': '.text-cnn_black_light3.text-sm',
                'tanggal': '.text-cnn_grey.text-sm.mb-4',
                'isi': '.detail-text p'
            },
            'liputan6.com': {
                'judul': '.read-page--header--title',
                'penulis': '.read-page--header--author__name',
                'tanggal': '.read-page--header--author__modified-time',
                'isi': '.article-content-body__item-content > p:not(.baca-juga):not([class]):not([id])'
            },
            'tribunnews.com': {
                'judul': '#arttitle',
                'penulis': '#penulis',
                'tanggal': 'time',
                'isi': '.txt-article p:not(.baca-juga):not([class]):not([id])'
            },
            
        }

    def scrape_berita(self, url, jenis_website):
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            selectors = self.selector_template.get(jenis_website, {})

            hasil = {}
            for key, selector in selectors.items():
                if key == 'isi':
                    elements = soup.select(selector)
                    hasil[key] = ' '.join([el.get_text(strip=True) for el in elements])
                else:
                    element = soup.select_one(selector)
                    hasil[key] = element.get_text(strip=True) if element else 'Tidak ditemukan'

            return hasil

        except Exception as e:
            return {'error': str(e)}

class TextSummarizer:
    def __init__(self):
        try:
            self.stopwords = set(stopwords.words('indonesian'))
        except:
            self.stopwords = {'yang', 'di', 'ke', 'dari', 'pada', 'dalam', 'untuk', 'dengan', 'dan', 'atau', 'juga', 'sebagai', 'adalah', 'oleh'}
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()

    def preprocess_text(self, text):
        removal_patterns = [
            r'Komik Si Calus.*',
            r'Loading\.\.\..*', 
            r'Ikuti Whatsapp Channel.*',
            r'sumber\s*:\s*Antara.*',
            r'Baca Juga:.*',
            r'REPUBLIKA\.CO\.ID.*?--',
        ]
        
        for pattern in removal_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def tokenisasi(self, text):
        return word_tokenize(text)

    def hapus_stopwords(self, words):
        return [word for word in words if word not in self.stopwords]

    def stemming(self, words):
        return [self.stemmer.stem(word) for word in words]

    def summarize_with_debug(self, text, persentase=0.3):
        kalimat = re.split(r'(?<=[.!?])\s+', text)  # Pisahkan berdasarkan kalimat
        preprocessed_kalimat = [self.preprocess_text(k) for k in kalimat]  # Preprocessing tiap kalimat
        tokens_kalimat = [self.tokenisasi(kalimat) for kalimat in preprocessed_kalimat]  # Tokenisasi
        tokens_tanpa_stopwords = [self.hapus_stopwords(tokens) for tokens in tokens_kalimat]  # Hapus stopwords
        tokens_stem = [self.stemming(tokens) for tokens in tokens_tanpa_stopwords]  # Stemming

        print("\n=== DEBUG: Preprocessing Kalimat ===")
        for i, k in enumerate(kalimat):
            print(f"Kalimat {i + 1}: {k}")
            print(f"   Preprocessed: {preprocessed_kalimat[i]}")
            print(f"   Tokens: {tokens_kalimat[i]}")
            print(f"   Tokens tanpa stopwords: {tokens_tanpa_stopwords[i]}")
            print(f"   Tokens setelah stemming: {tokens_stem[i]}")

        # Hitung IDF
        total_kalimat = len(tokens_stem)
        idf = {}
        for tokens in tokens_stem:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                idf[token] = idf.get(token, 0) + 1
        idf = {token: math.log(total_kalimat / freq) for token, freq in idf.items()}

        print("\n=== DEBUG: IDF Values ===")
        for token, value in idf.items():
            print(f"Token: {token}, IDF: {value}")

        # Hitung TF-IDF untuk setiap kalimat
        skor_kalimat = []
        debug_data = []
        for idx, tokens in enumerate(tokens_stem):
            tf = {token: tokens.count(token) for token in tokens}
            tf_idf = {token: tf[token] * idf[token] for token in tokens if token in idf}
            skor_total = sum(tf_idf.values())
            skor_kalimat.append((idx, skor_total))
            debug_data.append({
                'kalimat': kalimat[idx],
                'tokens': tokens,
                'tf': tf,
                'tf_idf': tf_idf,
                'skor': skor_total
            })

            print(f"\n=== DEBUG: TF-IDF untuk Kalimat {idx + 1} ===")
            print(f"Kalimat: {kalimat[idx]}")
            print(f"Tokens: {tokens}")
            print(f"TF: {tf}")
            print(f"TF-IDF: {tf_idf}")
            print(f"Skor Total: {skor_total}")

        # Urutkan berdasarkan skor TF-IDF
        skor_kalimat_terurut = sorted(skor_kalimat, key=lambda x: x[1], reverse=True)
        jumlah_kalimat = max(1, int(len(kalimat) * persentase))
        indeks_terpilih = sorted([idx for idx, _ in skor_kalimat_terurut[:jumlah_kalimat]])

        print("\n=== DEBUG: Seleksi Kalimat ===")
        for idx in indeks_terpilih:
            print(f"Kalimat terpilih: {idx + 1} - {kalimat[idx]}")

        # Kembalikan kalimat terpilih dan data debug
        ringkasan = ' '.join([kalimat[idx] for idx in indeks_terpilih])
        return {
            'ringkasan': ringkasan,
            'debug_data': debug_data
        }

@app.route('/scrape_berita', methods=['POST'])
def scrape_berita_endpoint():
    data = request.get_json()
    url = data.get('url')
    jenis_website = data.get('jenis_website')
    if not url or not jenis_website:
        return jsonify({'error': 'URL dan jenis website harus disertakan'}), 400
    scraper = BeritaScraper()
    hasil_scraping = scraper.scrape_berita(url, jenis_website)
    return jsonify(hasil_scraping)

@app.route('/cek_rouge', methods=['POST'])
def cek_rouge():
    data = request.json
    teks_asli = data.get('teks_asli', '')
    ringkasan = data.get('ringkasan', '')

    if not teks_asli or not ringkasan:
        return jsonify({'error': 'teks_asli dan ringkasan tidak boleh kosong!'}), 400

    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(teks_asli, ringkasan)
        rouge_scores = {
            'rouge1': {
                'f1': scores['rouge1'].fmeasure,
                'precision': scores['rouge1'].precision,
                'recall': scores['rouge1'].recall
            },
            'rouge2': {
                'f1': scores['rouge2'].fmeasure,
                'precision': scores['rouge2'].precision,
                'recall': scores['rouge2'].recall
            },
            'rougeL': {
                'f1': scores['rougeL'].fmeasure,
                'precision': scores['rougeL'].precision,
                'recall': scores['rougeL'].recall
            }
        }
        return jsonify(rouge_scores)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/summarize_berita', methods=['POST'])
def summarize_berita():
    data = request.get_json()
    url = data.get('url')
    jenis_website = data.get('jenis_website')
    persentase = data.get('persentase', 0.3)
    if not url or not jenis_website:
        return jsonify({'error': 'URL dan jenis website harus disertakan'}), 400
    scraper = BeritaScraper()
    hasil_scraping = scraper.scrape_berita(url, jenis_website)
    isi_berita = hasil_scraping.get('isi', '')
    if not isi_berita:
        return jsonify({'error': 'Isi berita tidak ditemukan'}), 400
    summarizer = TextSummarizer()
    hasil_ringkasan = summarizer.summarize_with_debug(isi_berita, persentase)
    hasil_scraping['ringkasan'] = hasil_ringkasan['ringkasan']
    return jsonify(hasil_scraping)

if __name__ == '__main__':
    app.run(debug=True)
