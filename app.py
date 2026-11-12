
import subprocess
import sys

try:
    import bert_score
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "bert-score"])
    import bert_score

import gradio as gr
import pandas as pd
import os
import torch
import time
import threading

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import search_engine
import summarizer
import analyzer

CSV_PATH = "/content/skripsi_app/data/data_klaster_gabungan.csv"
try:
    df_klaster = pd.read_csv(CSV_PATH)
    print("[INFO] Data CSV berhasil dimuat.")
except Exception as e:
    df_klaster = None
    print(f"[ERROR] Gagal memuat data CSV: {e}")

custom_css = """
body, .gradio-container { background-color: #0e1117 !important; color: #e0e0e0 !important; }
.gr-button-primary { background-color: #2e8b57 !important; border: none !important; color: white !important; font-weight: bold !important; transition: all 0.3s ease !important; border-radius: 8px !important; }
.gr-button-primary:hover { background-color: #3cb371 !important; transform: scale(1.02) !important; box-shadow: 0 4px 15px rgba(46,139,87,0.4) !important; }
.gr-button-secondary { background-color: transparent !important; border: 2px solid #2e8b57 !important; color: #2e8b57 !important; font-weight: bold !important; transition: all 0.3s ease !important; border-radius: 8px !important; }
.gr-button-secondary:hover { background-color: #2e8b57 !important; color: white !important; }
.gr-textbox, .gr-markdown { background-color: #1a1f2b !important; border: 1px solid #2e8b57 !important; border-radius: 8px !important; color: #e0e0e0 !important; }
.gr-textbox label, .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 { color: #3cb371 !important; }
@keyframes pulse { 0% { opacity: 0.6; } 50% { opacity: 1; } 100% { opacity: 0.6; } }
.progress-bar { animation: pulse 1.5s infinite !important; background-color: #2e8b57 !important; }
"""

def tutup_kalimat_utuh(teks):
    """Jika teks tidak diakhiri .!?, potong ke kalimat terakhir yang utuh."""
    if not teks:
        return teks
    # Cari titik terakhir
    for tanda in ['. ', '! ', '? ']:
        pos = teks.rfind(tanda)
        if pos > 50:  # minimal panjang teks
            # Pastikan setelah titik adalah spasi atau akhir
            if pos + 1 < len(teks) and teks[pos+1] == ' ':
                return teks[:pos+1]  # akhiri di titik (termasuk titik)
    # Jika tidak ditemukan, coba tanda tanpa spasi
    for tanda in ['.', '!', '?']:
        pos = teks.rfind(tanda)
        if pos > 50 and (pos == len(teks)-1 or teks[pos+1] == ' '):
            return teks[:pos+1]
    return teks  # kembalikan apa adanya kalau tidak ada

def proses_pencarian(kata_kunci):
    klaster, _ = search_engine.cari_klaster(kata_kunci, df_klaster)
    if klaster is None:
        return (None, "### Tidak ditemukan", gr.update(visible=False),
                gr.update(visible=False, value=""), gr.update(visible=False, value=""),
                gr.update(visible=False, value=""), gr.update(visible=False, value=""),
                gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False))
    lines = []
    lines.append("**Topik Klaster:** " + str(klaster.get('topik', 'Tidak diketahui')))
    lines.append("")
    for i in range(1, 6):
        judul = klaster.get(f'judul_{i}', f'Artikel {i}')
        isi = str(klaster.get(f'isi_{i}', ''))
        if pd.notna(judul) and isi.strip():
            # Potong kalimat terakhir yang tidak lengkap
            isi = tutup_kalimat_utuh(isi)
            if isi and isi[-1] not in '.!?':
                isi += '.'
            lines.append(f"<details><summary><b>{judul}</b></summary>")
            lines.append("")
            lines.append(isi)
            lines.append("")
            lines.append("</details>")
            lines.append("<hr>")
    md_artikel = "\n".join(lines)
    return (klaster, md_artikel, gr.update(visible=True),
            gr.update(visible=False, value=""), gr.update(visible=False, value=""),
            gr.update(visible=False, value=""), gr.update(visible=False, value=""),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False))

def proses_peringkasan(klaster):
    if klaster is None:
        return ("", "", "", "", {}, 0.0, 0.0, "")
    daftar_isi = [klaster.get(f'isi_{i}', '') for i in range(1, 6)]
    start_minilm = time.time()
    ringkasan_minilm, _ = summarizer.ringkas_berita("MiniLM", [], row=klaster)
    waktu_minilm = time.time() - start_minilm
    info_minilm = f"{waktu_minilm:.4f} detik"
    hasil_bart = {"ringkasan": "", "waktu": 0.0, "selesai": False}
    waktu_mulai_bart = time.time()
    def run_bart():
        start = time.time()
        ringkasan, _ = summarizer.ringkas_berita("BART", daftar_isi)
        hasil_bart["ringkasan"] = ringkasan
        hasil_bart["waktu"] = time.time() - start
        hasil_bart["selesai"] = True
    threading.Thread(target=run_bart).start()
    return (
        gr.update(visible=False, value=""),
        gr.update(visible=True, value=ringkasan_minilm),
        gr.update(visible=False, value="BART memproses... 0 detik"),
        gr.update(visible=True, value=info_minilm),
        hasil_bart, waktu_minilm, waktu_mulai_bart, ringkasan_minilm
    )

def update_info_bart(state_bart, waktu_mulai_bart, ringkasan_minilm, klaster, waktu_minilm):
    if state_bart.get("selesai", False):
        ringkasan_bart = state_bart["ringkasan"]
        info_bart = f"{state_bart['waktu']:.2f} detik"
        return (
            gr.update(visible=True, value=ringkasan_bart),
            gr.update(visible=True, value=info_bart),
            gr.update(visible=True, interactive=True),
            ringkasan_bart, state_bart["waktu"],
            gr.update(active=False)
        )
    else:
        elapsed = time.time() - waktu_mulai_bart
        return (
            gr.update(visible=False),
            gr.update(visible=False, value=f"BART memproses... {int(elapsed)} detik"),
            gr.update(visible=False, interactive=False),
            "", 0.0,
            gr.update(active=True)
        )

def proses_analisis(ringkasan_bart, ringkasan_minilm, klaster, waktu_bart, waktu_minilm):
    if not ringkasan_bart or not ringkasan_minilm:
        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
    analisis_bart = analyzer.analisis_berita(
        ringkasan_bart, klaster.get('ringkasan_referensi', None),
        perangkat="GPU" if torch.cuda.is_available() else "CPU",
        waktu_inferensi=waktu_bart, jumlah_param=610_000_000
    )
    analisis_minilm = analyzer.analisis_berita(
        ringkasan_minilm, klaster.get('ringkasan_referensi', None),
        perangkat="GPU" if torch.cuda.is_available() else "CPU",
        waktu_inferensi=waktu_minilm, jumlah_param=117_000_000
    )
    kesimpulan = "## Kesimpulan Perbandingan\n\n"
    r1_bart = r1_minilm = 0
    if klaster.get('ringkasan_referensi') and str(klaster.get('ringkasan_referensi')).strip().lower() != "nan":
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=False)
        r1_bart = scorer.score(str(klaster['ringkasan_referensi']), ringkasan_bart)['rouge1'].fmeasure * 100
        r1_minilm = scorer.score(str(klaster['ringkasan_referensi']), ringkasan_minilm)['rouge1'].fmeasure * 100
        kesimpulan += f"- **ROUGE-1 BART:** {r1_bart:.2f}%\n"
        kesimpulan += f"- **ROUGE-1 MiniLM:** {r1_minilm:.2f}%\n"
    kesimpulan += f"- **Waktu BART:** {waktu_bart:.2f} detik\n"
    kesimpulan += f"- **Waktu MiniLM:** {waktu_minilm:.3f} detik\n"
    if r1_bart > r1_minilm:
        kesimpulan += "\n**Kesimpulan:** BART unggul dalam kualitas, MiniLM unggul dalam kecepatan."
    else:
        kesimpulan += "\n**Kesimpulan:** MiniLM unggul dalam kualitas pada kasus ini, serta tetap unggul dalam kecepatan."
    return (
        gr.update(visible=True, value=analisis_bart),
        gr.update(visible=True, value=analisis_minilm),
        gr.update(visible=True, value=kesimpulan)
    )

with gr.Blocks(title="Prototipe Skripsi - NLP", css=custom_css) as app:
    state_klaster = gr.State()
    state_bart_result = gr.State({"selesai": False, "ringkasan": "", "waktu": 0.0})
    state_waktu_minilm = gr.State(0.0)
    state_waktu_mulai_bart = gr.State(0.0)
    state_ringkasan_bart = gr.State("")
    state_ringkasan_minilm = gr.State("")
    state_waktu_bart = gr.State(0.0)

    gr.Markdown("# Sistem Penyimpulan Berita Multi-Dokumen")
    gr.Markdown("Masukkan kata kunci, dapatkan ringkasan **BART** dan **MiniLM** secara berdampingan.")

    with gr.Row():
        with gr.Column(scale=3):
            input_kata_kunci = gr.Textbox(label="Kata Kunci Pencarian", placeholder="Ketik kata kunci...")
        with gr.Column(scale=1):
            btn_cari = gr.Button("Cari Berita", variant="primary")

    panel_artikel = gr.Markdown("### Artikel akan tampil di sini...")
    btn_ringkas = gr.Button("Ringkas 5 Artikel Ini", variant="primary", visible=False)

    with gr.Row():
        with gr.Column():
            output_bart = gr.Textbox(label="Ringkasan BART", lines=8, interactive=False, visible=False)
            info_bart = gr.Textbox(label="Waktu BART", visible=False)
        with gr.Column():
            output_minilm = gr.Textbox(label="Ringkasan MiniLM", lines=8, interactive=False, visible=False)
            info_minilm = gr.Textbox(label="Waktu MiniLM", visible=False)

    btn_analisis = gr.Button("Lihat Hasil Analisis", variant="secondary", visible=False, interactive=False)

    with gr.Row():
        with gr.Column():
            analisis_bart_box = gr.Markdown(visible=False)
        with gr.Column():
            analisis_minilm_box = gr.Markdown(visible=False)

    kesimpulan_box = gr.Markdown(visible=False)

    timer = gr.Timer(1, active=False)

    btn_cari.click(
        fn=proses_pencarian,
        inputs=[input_kata_kunci],
        outputs=[state_klaster, panel_artikel, btn_ringkas,
                 output_bart, output_minilm, info_bart, info_minilm,
                 btn_analisis, kesimpulan_box,
                 analisis_bart_box, analisis_minilm_box]
    )

    btn_ringkas.click(
        fn=proses_peringkasan,
        inputs=[state_klaster],
        outputs=[output_bart, output_minilm, info_bart, info_minilm,
                 state_bart_result, state_waktu_minilm, state_waktu_mulai_bart, state_ringkasan_minilm]
    ).then(
        fn=lambda: gr.update(active=True),
        outputs=[timer]
    )

    timer.tick(
        fn=update_info_bart,
        inputs=[state_bart_result, state_waktu_mulai_bart,
                state_ringkasan_minilm, state_klaster, state_waktu_minilm],
        outputs=[output_bart, info_bart, btn_analisis,
                 state_ringkasan_bart, state_waktu_bart,
                 timer]
    )

    btn_analisis.click(
        fn=proses_analisis,
        inputs=[state_ringkasan_bart, state_ringkasan_minilm, state_klaster,
                state_waktu_bart, state_waktu_minilm],
        outputs=[analisis_bart_box, analisis_minilm_box, kesimpulan_box]
    )

if __name__ == "__main__":
    app.launch(share=True)
