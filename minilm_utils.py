
import re
import time
import os
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

_MODEL_DIR = "/content/skripsi_app/model/minilm/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
minilm = SentenceTransformer(_MODEL_DIR)

def ambil_semua_kalimat(row):
    kalimat_total = []
    for j in range(1, 6):
        isi = str(row.get(f"isi_{j}", ""))
        if isi.strip():
            kalimat = nltk.sent_tokenize(isi)
            kalimat_total.extend(kalimat)
    return kalimat_total

def bersihkan_kalimat(teks):
    sampah = [
        r'SCROLL TO CONTINUE.*?CONTENT',
        r'SCROLL TO CONTINUE',
        r'BACA JUGA\s*:.*',
        r'Simak selengkapnya.*',
        r'\(?detikcom[^)]*\)?',
        r'Kompas\.com\s*[-–]\s*',
        r'Liputan6\.com\s*[-–]\s*',
        r'Reporter\s*:.*',
        r'Penulis\s*:.*',
        r'Editor\s*:.*',
        r'Foto\s*:.*',
        r'Ilustrasi\s*:.*',
        r'\[Gambar.*?\]',
        r'\(Sumber\s*:.*?\)',
    ]
    for pola in sampah:
        teks = re.sub(pola, '', teks, flags=re.IGNORECASE)
    # Hapus karakter non-ASCII
    teks = teks.encode('ascii', errors='ignore').decode('ascii')
    teks = re.sub(r'[•·▪▸►»>]', '', teks)
    # Ganti kutip ganda fancy menjadi kutip biasa
    teks = re.sub(r'[\u201c\u201d]', '"', teks)
    teks = re.sub(r'[()\[\]{}]', ' ', teks)
    teks = re.sub(r'(^|\s):\s+', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks

def bersihkan_final(teks):
    pembuka = [
        r'^Berikut\s+\w+\s+sajikan.*?[,:]?\s*',
        r'^Simak\s+.*?[,:]?\s*',
        r'^Dilansir\s+.*?[,:]?\s*',
        r'^Pantauan\s+.*?[,:]?\s*',
        r'^Dikutip\s+.*?[,:]?\s*',
        r'^Melansir\s+.*?[,:]?\s*',
        r'^Menurut\s+laporan\s+.*?[,:]?\s*',
    ]
    for pola in pembuka:
        teks = re.sub(pola, '', teks, flags=re.IGNORECASE)
    teks = re.sub(r'\s+', ' ', teks).strip()
    if teks and teks[0].islower():
        teks = teks[0].upper() + teks[1:]
    if teks and teks[-1] not in '.!?':
        teks += '.'
    return teks

def ringkas_minilm(row, top_k=7):
    kalimat = ambil_semua_kalimat(row)
    if not kalimat:
        return "", 0

    judul_set = {str(row.get(f"judul_{j}", "")).lower().strip() for j in range(1, 6)}
    kalimat = [bersihkan_kalimat(k) for k in kalimat]
    kalimat = [k for k in kalimat if len(k) >= 40 and k.lower().strip() not in judul_set]
    if not kalimat:
        return "", 0

    start = time.time()
    emb_kalimat = minilm.encode(kalimat)
    emb_kunci = minilm.encode([row['topik']])
    similarities = cosine_similarity(emb_kunci, emb_kalimat)[0]

    idx_urut = similarities.argsort()[::-1]
    kalimat_terpilih = []
    total_kata = 0

    for idx in idx_urut:
        if similarities[idx] < 0.2:
            continue
        k = kalimat[idx]
        kata_dalam_kalimat = len(k.split())
        if total_kata + kata_dalam_kalimat <= 200:
            kalimat_terpilih.append((idx, k))
            total_kata += kata_dalam_kalimat
        if total_kata >= 150:
            break

    durasi = time.time() - start
    if not kalimat_terpilih:
        return "", durasi

    kalimat_terpilih.sort(key=lambda x: x[0])
    ringkasan = " ".join([k for _, k in kalimat_terpilih])
    ringkasan = bersihkan_final(ringkasan)
    return ringkasan, durasi
