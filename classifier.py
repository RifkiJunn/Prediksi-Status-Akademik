"""Klasifikasi status akademik mahasiswa menggunakan LLM.
Mendukung beberapa AI service yang kompatibel dengan format OpenAI (Groq & OpenRouter),
sehingga kode evaluasi yang sama bisa dipakai untuk membandingkan layanan."""
import json
import os
import re
import time

import openai
from openai import OpenAI

# daftar AI service: base URL + nama variabel API key di file .env
PROVIDER = {
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "env_key": "GEMINI_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "env_key": "OPENROUTER_API_KEY"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "env_key": "GROQ_API_KEY"},
}
MODEL_DEFAULT = {
    "gemini": "gemini-3.1-flash-lite",            # Google AI Studio (free tier)
    "groq": "llama-3.1-8b-instant",               # model bawaan Modul 2A
    "openrouter": "google/gemma-4-31b-it:free",   # model gratis di OpenRouter
}

# nama fitur yang ditampilkan ke LLM
NAMA_FITUR = {
    "ipk": "IPK",
    "kehadiran": "Kehadiran (%)",
    "jam_belajar": "Jam belajar mandiri per minggu",
    "tugas_terkumpul": "Tugas terkumpul (%)",
    "nilai_uts": "Nilai UTS",
    "aktif_organisasi": "Aktif organisasi",
}


def buat_client(provider):
    p = PROVIDER[provider]
    return OpenAI(base_url=p["base_url"], api_key=os.environ[p["env_key"]])


def load_config(path="config.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_data(data):
    """dict data mahasiswa -> teks satu baris untuk LLM."""
    return "; ".join(f"{nama}: {data[kolom]}" for kolom, nama in NAMA_FITUR.items())


def build_messages(cfg, data):
    msgs = [{"role": "system", "content": cfg["system_prompt"]}]
    for ex in cfg.get("few_shot", []):                      # contoh (few-shot), boleh kosong
        msgs.append({"role": "user", "content": format_data(ex["data"])})
        msgs.append({"role": "assistant", "content": ex["label"]})
    msgs.append({"role": "user", "content": format_data(data)})
    return msgs


def parse_label(output, labels):
    """Rapikan jawaban model: 'waspada.' -> 'Waspada'. Di luar daftar label -> 'invalid'."""
    jawab = re.sub(r"[^a-z]", "", (output or "").lower())
    peta = {label.lower(): label for label in labels}
    if jawab in peta:
        return peta[jawab]
    # toleransi: jawaban seperti "Status: Waspada" -> diterima jika tepat satu label disebut
    kata = set(re.findall(r"[a-z]+", (output or "").lower()))
    cocok = [peta[k] for k in peta if k in kata]
    return cocok[0] if len(cocok) == 1 else "invalid"


class KuotaHabis(Exception):
    """Batas request terus-menerus tercapai (mis. kuota harian free tier habis)."""


def classify(client, cfg, data, max_retry=4):
    """Kembalikan label, atau 'invalid' bila jawaban model di luar daftar label.
    Jika gagal memanggil API berkali-kali, lempar KuotaHabis agar evaluasi berhenti
    (bukan diam-diam dihitung sebagai prediksi salah)."""
    for percobaan in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=cfg["model"],
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
                messages=build_messages(cfg, data),
            )
            return parse_label(resp.choices[0].message.content, cfg["labels"])
        except openai.RateLimitError:
            time.sleep(15 * (percobaan + 1))                # kena batas kuota: tunggu lalu coba lagi
        except (openai.APIConnectionError, openai.InternalServerError, IndexError, TypeError):
            time.sleep(5)                                   # gangguan server/respons kosong: coba lagi
    raise KuotaHabis(f"Gagal memanggil {cfg['model']} setelah {max_retry} percobaan")
