import re
import difflib
import pandas as pd

def cari_klaster(kata_kunci: str, df: pd.DataFrame):
    """
    Mencari klaster berita terbaik berdasarkan kata kunci pada 5 kolom judul.
    
    Akurasi dijamin dengan pencarian kata utuh (regex \\b) untuk menghindari
    kesalahan substring (contoh: 'iran' tercari di dalam 'khawatiran').
    """
    if df is None or df.empty:
        return None, "Data CSV kosong atau belum dimuat."
    
    kata_kunci = kata_kunci.strip().lower()
    if not kata_kunci:
        return None, "Kata kunci kosong."

    best_cluster = None
    max_score = 0
    
    # Buat pola regex kata utuh (case-insensitive) untuk akurasi tinggi
    pattern = re.compile(rf'\b{re.escape(kata_kunci)}\b', re.IGNORECASE)
    
    # Daftar 5 kolom judul sesuai struktur data_klaster_gabungan.csv
    kolom_judul = [f'judul_{i}' for i in range(1, 6)]
    
    # --- TAHAP 1: Pencarian Eksak (Kata Utuh) ---
    for _, row in df.iterrows():
        score = 0
        for col in kolom_judul:
            if col in df.columns and pd.notna(row[col]):
                # Hitung jumlah judul dalam klaster yang mengandung kata utuh
                if pattern.search(str(row[col])):
                    score += 1
                    
        if score > max_score:
            max_score = score
            best_cluster = row
            
    # --- TAHAP 2: Fallback ke Fuzzy Matching (Jika Eksak = 0) ---
    if max_score == 0:
        highest_ratio = 0.0
        for _, row in df.iterrows():
            # Gabungkan 5 judul untuk dicocokkan kemiripannya
            gabungan_judul = " ".join([
                str(row[col]).lower() 
                for col in kolom_judul 
                if col in df.columns and pd.notna(row[col])
            ])
            
            # Cari rasio kemiripan (threshold/cutoff = 0.5)
            ratio = difflib.SequenceMatcher(None, kata_kunci, gabungan_judul).ratio()
            if ratio >= 0.5 and ratio > highest_ratio:
                highest_ratio = ratio
                best_cluster = row
                
        if best_cluster is not None:
            return best_cluster, f"Klaster ditemukan via Fuzzy Matching (Kemiripan: {highest_ratio:.2%})"
        else:
            return None, "Tidak ada klaster berita yang cocok dengan kata kunci tersebut."
            
    return best_cluster, f"Klaster ditemukan via Pencarian Eksak ({max_score} judul cocok)"