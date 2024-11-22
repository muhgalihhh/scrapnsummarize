import csv

import pandas as pd
import requests
from bs4 import BeautifulSoup


class DetikScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def scrape_article(self, url):
        """
        Scrape artikel dari Detik.com
        
        :param url: URL artikel Detik.com
        :return: Dictionary berisi informasi artikel
        """
        try:
            # Fetch halaman
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Ekstrak judul
            judul = soup.select_one('.detail__title') 
            judul = judul.get_text(strip=True) if judul else 'Tidak ada judul'
            
            # Ekstrak penulis
            penulis = soup.select_one('.detail__author') 
            penulis = penulis.get_text(strip=True) if penulis else 'Tidak diketahui'
            
            # Ekstrak tanggal
            tanggal = soup.select_one('.detail__date') 
            tanggal = tanggal.get_text(strip=True) if tanggal else 'Tidak ada tanggal'
            
            # Ekstrak isi artikel
            paragraf = soup.select('.detail__body-text p')
            isi_artikel = ' '.join([p.get_text(strip=True) for p in paragraf])
            
            return {
                'Judul': judul,
                'Penulis': penulis,
                'Tanggal': tanggal,
                'Isi Artikel': isi_artikel
            }
        
        except requests.RequestException as e:
            print(f"Error fetching page: {e}")
            return None
    
    def save_to_csv(self, data, filename='detik_artikel.csv'):
        """
        Simpan data ke file CSV
        
        :param data: Dictionary data artikel
        :param filename: Nama file output
        """
        if not data:
            print("Tidak ada data untuk disimpan.")
            return
        
        with open(filename, 'w', newline='', encoding='utf-8') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=data.keys())
            writer.writeheader()
            writer.writerow(data)
        
        print(f"Data berhasil disimpan ke {filename}")
    
    def save_to_excel(self, data, filename='detik_artikel.xlsx'):
        """
        Simpan data ke file Excel
        
        :param data: Dictionary data artikel
        :param filename: Nama file output
        """
        if not data:
            print("Tidak ada data untuk disimpan.")
            return
        
        df = pd.DataFrame([data])
        df.to_excel(filename, index=False)
        print(f"Data berhasil disimpan ke {filename}")

def main():
    # URL artikel Detik.com yang ingin di-scrape
    url = 'https://www.detik.com/properti/berita/d-7650019/menengok-rumah-korban-gempa-cianjur-hampir-seluruh-unit-terisi'
    
    # Inisialisasi scraper
    scraper = DetikScraper()
    
    # Scrape artikel
    artikel = scraper.scrape_article(url)
    
    # Simpan data
    if artikel:
        scraper.save_to_csv(artikel)
        scraper.save_to_excel(artikel)

if __name__ == '__main__':
    main()
