import math
import re

import nltk
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer  # Mengimpor PorterStemmer
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt')
nltk.download('all')
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
                'isi': '#article_content > p:not(.baca):not([class]):not([id])'
            },
            'republika.co.id': {
                'judul': '.max-card__title h1, .max-card__titlep',
                'penulis': '.max-card__title a',
                'tanggal': '.date',
                'isi': '.article-content p:not(.premium-content):not([class]):not([id])'
            },
            'sindonews.com': {
                'judul': '.detail-title',
                'penulis': '.detail-nama-redaksi',
                'tanggal': '.detail-date-artikel',
                'isi': '.detail-desc > p:not([class]):not([id])'
            },
            'okezone.com': {
                'judul': '.title h1',
                'penulis': '.reporter .namerep a',
                'tanggal': '.reporter .namerep b',
                'isi': '#contentx > p:not(.baca-juga):not(#bacajuga):not([class]):not([id])'
            },
            'suara.com': {
                'judul': '.info h1',
                'penulis': '.writer a',
                'tanggal': '.date',
                'isi': '.detail-content > p:not(.baca-juga-new):not([class]):not([id])'
            },
            'idntimes.com': {
                'judul': '.title-text',
                'penulis': '.author-name a',
                'tanggal': '.date',
                'isi': '#article-description p'
            },
            'merdeka.com': {
                'judul': '.article-title',
                'penulis': '.dt--postcredit-editor-desc',
                'tanggal': 'time span',
                'isi': '.article p'
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
        self.stemmer = PorterStemmer()  # Inisialisasi stemmer

    def preprocess_text(self, text):
        # Daftar pola untuk dihapus
        removal_patterns = [
            r'Komik Si Calus.*',  # Hapus bagian komik
            r'Loading\.\.\..*',  # Hapus bagian loading
            r'Ikuti Whatsapp Channel.*',  # Hapus bagian channel
            r'sumber\s*:\s*Antara.*',  # Hapus sumber berita
            r'Baca Juga:.*',  # Hapus bagian "Baca Juga"
            r'REPUBLIKA\.CO\.ID.*?--',  # Hapus header/metadata awal
        ]
        
        # Hapus pola-pola yang tidak diinginkan
        for pattern in removal_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Normalisasi dan bersihkan teks
        text = re.sub(r'[^\w\s]', '', text)  # Hapus karakter non-alfanumerik
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()  # Normalisasi spasi
        
        words = word_tokenize(text)
        words = [word for word in words if word not in self.stopwords]
        
        # Terapkan stemming pada kata-kata
        words = [self.stemmer.stem(word) for word in words]
        
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
    hasil_scraping = scraper.scrape_berita(url, jenis_website)
    if 'isi' not in hasil_scraping:
        return jsonify({'error': 'Isi artikel tidak ditemukan'}), 400

    summarizer = TextSummarizer()
    ringkasan = summarizer.summarize(hasil_scraping['isi'], persentase)
    
    return jsonify({'judul': hasil_scraping['judul'], 'ringkasan': ringkasan})

if __name__ == '__main__':
    app.run(debug=True)
