import time

import pandas as pd
import streamlit as st
from openai import OpenAI

from classifier import NAMA_FITUR, PROVIDER, KuotaHabis, classify, load_config

st.set_page_config(page_title="Status Akademik (LLM)", page_icon=":material/school:")

# warna & arti tiap status (definisi sama dengan system prompt)
STATUS = {
    "Aman": ("#1B7A4B", "#E6F4EC", "Performa akademik baik secara keseluruhan."),
    "Waspada": ("#9A6700", "#FFF4D6", "Performa sedang, ada indikator yang perlu diperhatikan."),
    "Berisiko": ("#B42318", "#FDECEA", "Performa rendah, berisiko gagal studi."),
}
st.markdown("""<style>
.block-container {max-width: 880px; padding-top: 2.5rem;}
h1 {letter-spacing: -0.03em; font-weight: 800;}
::selection {background: #1F4E8C; color: #fff;}
.hasil {border-radius: 14px; padding: 1.4rem 1.6rem; margin-top: .5rem;
        box-shadow: 0 6px 20px rgba(20,33,61,.08); animation: muncul .35s cubic-bezier(.16,1,.3,1);}
.hasil small {text-transform: uppercase; letter-spacing: .06em; font-weight: 600; opacity: .8;}
.hasil b {display: block; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em; margin: .1rem 0 .3rem;}
@keyframes muncul {from {opacity: .4; transform: translateY(6px);} to {opacity: 1; transform: none;}}
</style>""", unsafe_allow_html=True)


def kartu_hasil(label):
    fg, bg, arti = STATUS.get(label, ("#14213D", "#EAF0F7", "Jawaban model di luar daftar label."))
    st.markdown(f'<div class="hasil" style="background:{bg};color:{fg}">'
                f'<small>Prediksi status</small><b>{label}</b>{arti}</div>', unsafe_allow_html=True)

# ---------- 1. Muat konfigurasi hasil evaluasi & siapkan client ----------
cfg = load_config("config.json")
FITUR = list(NAMA_FITUR)
PROV = cfg["provider"]                          # "gemini" (hasil Modul 2A)
NAMA_KEY = PROVIDER[PROV]["env_key"]            # GEMINI_API_KEY

try:
    API_KEY = st.secrets[NAMA_KEY]
except Exception:                       # file secrets belum ada atau key belum diisi
    st.error(f"{NAMA_KEY} belum diatur. Isi di .streamlit/secrets.toml (lokal) "
             "atau menu Secrets (Streamlit Community Cloud).")
    st.stop()


@st.cache_resource
def get_client():
    return OpenAI(base_url=PROVIDER[PROV]["base_url"], api_key=API_KEY)


client = get_client()


@st.cache_data(show_spinner=False)
def prediksi(ipk, kehadiran, jam_belajar, tugas_terkumpul, nilai_uts, aktif_organisasi):
    # hasil disimpan di cache: data yang sama tidak memanggil API dua kali (hemat kuota)
    data = {"ipk": ipk, "kehadiran": kehadiran, "jam_belajar": jam_belajar,
            "tugas_terkumpul": tugas_terkumpul, "nilai_uts": nilai_uts,
            "aktif_organisasi": aktif_organisasi}
    return classify(client, cfg, data)


# ---------- 2. Sidebar: informasi konfigurasi yang dikunci ----------
ev = cfg["evaluasi"]
with st.sidebar:
    st.subheader(":material/insights: Tentang Model")
    with st.container(border=True):
        st.caption("Model")
        st.markdown(f"**{cfg['model']}**  \n{PROV.capitalize()} · {len(cfg['few_shot'])} contoh few-shot")
        lolos = ev["f1_macro"] >= 0.70
        st.badge("Lolos gerbang evaluasi" if lolos else "Belum lolos gerbang (F1 < 0,70)",
                 icon=":material/check_circle:" if lolos else ":material/error:",
                 color="green" if lolos else "orange")

    st.caption(f"Kinerja pada {ev['n']} data uji (sama dengan Modul 1A)")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{ev['accuracy']:.2f}")
    if "f1_dnn_pembanding" in ev:
        c2.metric("F1 macro", f"{ev['f1_macro']:.2f}",
                  delta=f"{ev['f1_macro'] - ev['f1_dnn_pembanding']:+.2f} vs DNN")
        st.progress(ev["f1_macro"], text=f"LLM  {ev['f1_macro']:.3f}")
        st.progress(ev["f1_dnn_pembanding"], text=f"DNN Modul 1A  {ev['f1_dnn_pembanding']:.3f}")
    else:
        c2.metric("F1 macro", f"{ev['f1_macro']:.2f}")

    st.caption("Arti status")
    for s, (fg, bg, arti) in STATUS.items():
        st.markdown(f'<div style="background:{bg};color:{fg};border-radius:10px;padding:.45rem .7rem;'
                    f'margin-bottom:.35rem;font-size:.85rem"><b>{s}</b> · {arti}</div>',
                    unsafe_allow_html=True)

    with st.expander(":material/description: Lihat system prompt"):
        st.code(cfg["system_prompt"], language=None, wrap_lines=True)
    st.caption(f"Dievaluasi {ev['tanggal']} · konfigurasi dikunci dari Modul 2A")

st.title("Prediksi Status Akademik")
st.caption(f"Klasifikasi dengan LLM `{cfg['model']}` · hasil: **Aman**, **Waspada**, atau **Berisiko**")

tab1, tab2 = st.tabs([":material/person: Satu Mahasiswa", ":material/table: Dari File CSV"])

# ---------- 3. Prediksi satu data ----------
with tab1:
    with st.form("form_prediksi", border=True):
        col1, col2 = st.columns(2)
        with col1:
            ipk = st.number_input("IPK", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
            kehadiran = st.slider("Kehadiran (%)", 0, 100, 80)
            jam_belajar = st.slider("Jam belajar mandiri per minggu", 0, 40, 8)
        with col2:
            tugas = st.slider("Tugas terkumpul (%)", 0, 100, 80)
            uts = st.slider("Nilai UTS", 0, 100, 70)
            organisasi = st.radio("Aktif organisasi?", ["Ya", "Tidak"], horizontal=True)
        kirim = st.form_submit_button("Prediksi status", type="primary", icon=":material/psychology:",
                                      use_container_width=True)

    if kirim:
        with st.spinner(f"Menghubungi {PROV.capitalize()}..."):
            try:
                hasil = prediksi(ipk, kehadiran, jam_belajar, tugas, uts, organisasi)
            except KuotaHabis:
                st.error("Kuota API sedang habis atau server sibuk. Coba lagi beberapa saat lagi.")
                st.stop()
        kartu_hasil(hasil)

# ---------- 4. Prediksi banyak data dari CSV ----------
with tab2:
    st.write("Unggah CSV dengan kolom: " + ", ".join(f"`{c}`" for c in FITUR)
             + " (maksimal 30 baris agar hemat kuota).")
    file = st.file_uploader("Pilih file CSV", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        kurang = [c for c in FITUR if c not in df.columns]
        if kurang:
            st.error(f"Kolom berikut tidak ditemukan: {', '.join(kurang)}")
        else:
            df = df.head(30)
            st.dataframe(df, hide_index=True)
            if st.button(f"Prediksi {len(df)} mahasiswa", type="primary", icon=":material/psychology:"):
                bar = st.progress(0.0)
                hasil = []
                for i, (_, b) in enumerate(df.iterrows(), start=1):
                    try:
                        hasil.append(prediksi(float(b["ipk"]), int(b["kehadiran"]), int(b["jam_belajar"]),
                                              int(b["tugas_terkumpul"]), int(b["nilai_uts"]),
                                              str(b["aktif_organisasi"])))
                    except KuotaHabis:
                        hasil.append("gagal (kuota)")
                    bar.progress(i / len(df))
                    time.sleep(5)               # jaga batas request per menit
                df["prediksi_status"] = hasil
                jumlah = df["prediksi_status"].value_counts()
                for kol, s in zip(st.columns(3), STATUS):
                    kol.metric(s, int(jumlah.get(s, 0)))
                warna = lambda v: f"color:{STATUS[v][0]};background:{STATUS[v][1]};font-weight:600" if v in STATUS else ""
                st.dataframe(df.style.map(warna, subset=["prediksi_status"]), hide_index=True)
                st.download_button("Unduh hasil prediksi", df.to_csv(index=False),
                                   file_name="hasil_prediksi_llm.csv", mime="text/csv")
