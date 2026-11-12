import os
import torch
import re
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from minilm_utils import ringkas_minilm

# -------------------------------------------------------------------------
# SETUP PERANGKAT
# -------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------------------
# 1. KONFIGURASI MODEL BART
# -------------------------------------------------------------------------
BART_PATH = "model/bart"
tokenizer_bart = None
model_bart = None

def load_bart():
    global tokenizer_bart, model_bart
    if model_bart is None:
        print(f"[{device.upper()}] Memuat model mBART dari {BART_PATH}...")
        tokenizer_bart = MBart50TokenizerFast.from_pretrained(
            BART_PATH, src_lang="id_ID", tgt_lang="id_ID"
        )
        model_bart = MBartForConditionalGeneration.from_pretrained(BART_PATH).to(device)
    return tokenizer_bart, model_bart

# -------------------------------------------------------------------------
# 2. PEMBERSIHAN TEKS
# -------------------------------------------------------------------------
def bersihkan_teks(teks):
    teks = re.sub(r'\[JUDUL\]|\[SEP\]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    if teks and not teks.endswith('.'):
        teks += '.'
    return teks

# -------------------------------------------------------------------------
# 3. FUNGSI UTAMA PERINGKASAN (BART & MINILM)
# -------------------------------------------------------------------------
def ringkas_berita(pilihan_model: str, daftar_isi: list, row=None):
    """
    pilihan_model : "BART" atau "MiniLM"
    daftar_isi    : list isi artikel (untuk BART)
    row           : Series/DataFrame row klaster (untuk MiniLM)
    """
    if pilihan_model == "BART":
        tok, mod = load_bart()

        # --- TAHAP 1: Ringkas Per Artikel ---
        ringkasan_tahap1 = []
        for teks in daftar_isi:
            if not teks or str(teks).strip().lower() in ("", "nan"):
                continue
            inputs = tok(str(teks), return_tensors="pt",
                         max_length=1024, truncation=True).to(device)
            summary_ids = mod.generate(
                inputs["input_ids"],
                max_length=80, min_length=25,
                num_beams=3, early_stopping=True
            )
            ringkasan_tahap1.append(tok.decode(summary_ids[0], skip_special_tokens=True))

        if not ringkasan_tahap1:
            return "Tidak ada teks yang valid untuk diringkas.", device.upper()

        teks_gabungan = " [SEP] ".join(ringkasan_tahap1)
        inputs_tahap2 = tok(teks_gabungan, return_tensors="pt",
                            max_length=1024, truncation=True).to(device)

        summary_ids_final = mod.generate(
            inputs_tahap2["input_ids"],
            max_length=600, min_length=250,
            num_beams=4, no_repeat_ngram_size=3,
            length_penalty=1.5, repetition_penalty=1.2,
            early_stopping=False
        )
        hasil_mentah = tok.decode(summary_ids_final[0], skip_special_tokens=True)
        return bersihkan_teks(hasil_mentah), device.upper()

    elif pilihan_model == "MiniLM":
        if row is None:
            return "Data klaster tidak tersedia untuk MiniLM.", device.upper()
        ringkasan, _ = ringkas_minilm(row)
        return ringkasan, device.upper()

    return "Model tidak valid.", device.upper()