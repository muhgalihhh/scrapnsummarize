import math
import re

import nltk
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

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
                'isi': '.detail__body-text  p:not(.para_caption)'
            },
            'kompas.com': {
                'judul': '.read__title',
                'penulis': '.read__author',
                'tanggal': '.read__time',
                'isi': '.read__content p:not(:has(strong))'
            }
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

            print(f"Hasil scraping: {hasil}")  # Debug hasil scraping
            return hasil

        except Exception as e:
            return {'error': str(e)}

class TextSummarizer:
    def __init__(self):
        try:
            self.stopwords = set(stopwords.words('indonesian'))
        except:
            self.stopwords = {'yang', 'di', 'ke', 'dari', 'pada', 'dalam', 'untuk', 'dengan'}

    def preprocess_text(self, text):
        print(f"Teks asli: {text}")  # Debug teks asli
        
        # Bersihkan teks dari pola-pola iklan
        text = re.sub(r'ADVERTISEMENT.*?CONTENT', '', text, flags=re.IGNORECASE)
        text = re.sub(r'SCROLL TO CONTINUE.*?$', '', text, flags=re.IGNORECASE)

        # Lanjutkan ke langkah preprocessing lainnya
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hanya huruf dan spasi
        words = word_tokenize(text)
        words = [word for word in words if word not in self.stopwords]
        print(f"Hasil preprocessing: {words}")  # Debug hasil preprocessing
        return words


    def tokenize_sentences(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        print(f"Hasil tokenisasi kalimat: {sentences}")  # Debug hasil tokenisasi kalimat
        return sentences

    def hitung_idf(self, documents):
        idf = {}
        total_doc = len(documents)
        doc_frekuensi = {}
        for doc in documents:
            unique_words = set(doc)
            for word in unique_words:
                doc_frekuensi[word] = doc_frekuensi.get(word, 0) + 1
        for word, freq in doc_frekuensi.items():
            idf[word] = math.log(total_doc / (freq + 1))
        print(f"Hasil IDF: {idf}")  # Debug hasil IDF
        return idf

    def summarize(self, text, persentase=0.3):
        kalimat = self.tokenize_sentences(text)
        preprocessed_kalimat = [self.preprocess_text(k) for k in kalimat]
        idf = self.hitung_idf(preprocessed_kalimat)
        skor_kalimat = [(idx, sum(idf.get(word, 0) for word in words)) for idx, words in enumerate(preprocessed_kalimat)]
        print(f"Skor setiap kalimat: {skor_kalimat}")  # Debug skor setiap kalimat
        skor_kalimat_terurut = sorted(skor_kalimat, key=lambda x: x[1], reverse=True)
        jumlah_kalimat = max(1, int(len(kalimat) * persentase))
        indeks_terpilih = sorted([idx for idx, _ in skor_kalimat_terurut[:jumlah_kalimat]])
        print(f"Kalimat terpilih berdasarkan skor: {indeks_terpilih}")  # Debug kalimat terpilih
        return ' '.join([kalimat[idx] for idx in indeks_terpilih])

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

@app.route('/summarize_berita', methods=['POST'])
def summarize_berita():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON payload'}), 400
    url = data.get('url')
    jenis_website = data.get('jenis_website')
    persentase = data.get('persentase', 0.3)
    if not url or not jenis_website:
        return jsonify({'error': 'URL dan jenis website harus disertakan'}), 400
    scraper = BeritaScraper()
    artikel = scraper.scrape_berita(url, jenis_website)
    if 'error' in artikel:
        return jsonify(artikel), 400
    summarizer = TextSummarizer()
    ringkasan = summarizer.summarize(artikel['isi'], persentase)
    return jsonify({'ringkasan': ringkasan, 'judul_asli': artikel.get('judul', '')})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
