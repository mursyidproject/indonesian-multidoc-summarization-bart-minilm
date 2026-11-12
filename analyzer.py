
import re
from rouge_score import rouge_scorer
from bert_score import BERTScorer

try:
    bert_scorer = BERTScorer(lang="id", model_type="distilbert-base-multilingual-cased")
except Exception as e:
    bert_scorer = None
    print(f"[WARN] BERTScore tidak dapat dimuat: {e}")

def analisis_berita(ringkasan_sistem, ringkasan_referensi=None, perangkat="CPU",
                    waktu_inferensi=0.0, jumlah_param=0):
    if not ringkasan_sistem or ringkasan_sistem.strip() == "":
        return "⚠️ Tidak ada ringkasan yang dievaluasi."

    kata = ringkasan_sistem.split()
    jumlah_kata = len(kata)

    hasil = "### 📊 Analisis Ringkasan\n"
    hasil += f"- **Panjang**: {jumlah_kata} kata\n"

    # ROUGE
    if ringkasan_referensi and str(ringkasan_referensi).strip().lower() != "nan":
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        scores = scorer.score(str(ringkasan_referensi), ringkasan_sistem)
        hasil += f"- **ROUGE-1** (kesamaan kata tunggal): {scores['rouge1'].fmeasure*100:.2f}%\n"
        hasil += f"- **ROUGE-2** (kesamaan pasangan kata): {scores['rouge2'].fmeasure*100:.2f}%\n"
        hasil += f"- **ROUGE-L** (kesamaan struktur kalimat): {scores['rougeL'].fmeasure*100:.2f}%\n"
    else:
        hasil += "- **ROUGE**: ⚠️ Tidak tersedia (tidak ada referensi)\n"

    # BERTScore
    if ringkasan_referensi and str(ringkasan_referensi).strip().lower() != "nan" and bert_scorer is not None:
        try:
            P, R, F1 = bert_scorer.score([ringkasan_sistem], [str(ringkasan_referensi)])
            hasil += f"- **BERTScore Precision** (ketepatan makna): {P.mean().item():.4f}\n"
            hasil += f"- **BERTScore Recall** (kelengkapan makna): {R.mean().item():.4f}\n"
            hasil += f"- **BERTScore F1** (keseimbangan makna): {F1.mean().item():.4f}\n"
        except Exception as e:
            hasil += f"- **BERTScore**: ⚠️ Gagal dihitung ({str(e)})\n"
    elif bert_scorer is None:
        hasil += "- **BERTScore**: ⚠️ Model tidak tersedia\n"
    else:
        hasil += "- **BERTScore**: ⚠️ Tidak tersedia (tidak ada referensi)\n"

    if waktu_inferensi > 0:
        hasil += f"- **Waktu Inferensi**: {waktu_inferensi:.3f} detik\n"
    if jumlah_param > 0:
        hasil += f"- **Parameter**: {jumlah_param/1e6:.0f} juta\n"
    hasil += f"- **Perangkat**: {perangkat}\n"

    return hasil
