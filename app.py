import streamlit as st
from huggingface_hub import InferenceClient
import time

# --- SETUP HALAMAN ---
st.set_page_config(page_title="Qwen AI Generator", page_icon="🤖", layout="centered")

# --- AMBIL API KEY DARI SECRETS (BRANKAS RAHASIA) ---
try:
    # Ini kuncinya! App akan mencari variabel bernama "HF_TOKEN" di server
    api_token = st.secrets["HF_TOKEN"]
except FileNotFoundError:
    st.error("⚠️ API Token belum disetting! Mohon atur di Streamlit Secrets.")
    st.stop()

# --- KONFIGURASI MODEL ---
model_id = "Qwen/Qwen2.5-7B-Instruct"

# --- FUNGSI GENERATE (CHAINING) ---
def get_response(messages):
    """Fungsi kirim pesan ke Hugging Face"""
    client = InferenceClient(model=model_id, token=api_token)
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=1500,  # Panjang maksimum per respons
            temperature=0.7,  # Tingkat kreativitas
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# --- TAMPILAN UI ---
st.title("🤖 Qwen 2.5 Article Writer")
st.markdown("Generate artikel panjang otomatis tanpa ribet input API Key.")

with st.form("input_form"):
    topic = st.text_input("Topik Artikel", placeholder="Contoh: Manfaat AI untuk Bisnis UMKM")
    col1, col2 = st.columns(2)
    with col1:
        tone = st.selectbox("Gaya Bahasa", ["Informatif", "Santai", "Persuasif", "Formal"])
    with col2:
        lang = st.selectbox("Bahasa", ["Bahasa Indonesia", "English"])
    
    submit_btn = st.form_submit_button("🚀 Mulai Menulis")

# --- LOGIKA UTAMA ---
if submit_btn and topic:
    st.info("Sedang bekerja... (Proses ini memakan waktu +/- 30 detik)")
    
    # 1. BUAT OUTLINE
    status_text = st.empty()
    status_text.markdown("**Langkah 1/2:** Membuat kerangka tulisan...")
    
    prompt_outline = f"Buatkan outline artikel tentang '{topic}'. Gaya: {tone}. Bahasa: {lang}. Format: Hanya list poin utama (1, 2, 3, dst). Jangan ada teks lain."
    outline = get_response([{"role": "user", "content": prompt_outline}])
    
    if "Error" in outline:
        st.error(outline)
    else:
        # Bersihkan outline jadi list
        sections = [s.strip() for s in outline.split('\n') if s.strip() and (s[0].isdigit() or s.startswith('-'))]
        
        full_article = f"# {topic}\n\n"
        
        # 2. TULIS PER BAGIAN (LOOPING)
        progress_bar = st.progress(0)
        
        for i, section in enumerate(sections):
            status_text.markdown(f"**Langkah 2/2:** Menulis bagian: *{section}*...")
            
            prompt_content = f"Tuliskan isi paragraf lengkap untuk sub-judul: '{section}'. Topik utama: {topic}. Gaya: {tone}. Bahasa: {lang}. Panjang minimal 2 paragraf."
            content = get_response([{"role": "user", "content": prompt_content}])
            
            full_article += f"## {section}\n{content}\n\n"
            progress_bar.progress((i + 1) / len(sections))
            time.sleep(1) # Jeda sedikit biar tidak kena limit
            
        status_text.success("✅ Artikel Selesai!")
        
        # TAMPILKAN HASIL
        st.markdown("---")
        st.markdown(full_article)
        st.download_button("Unduh Artikel (.txt)", full_article, file_name="artikel.txt")

elif submit_btn and not topic:
    st.warning("Mohon isi topik artikel terlebih dahulu.")
