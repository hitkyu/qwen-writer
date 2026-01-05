import streamlit as st
from huggingface_hub import InferenceClient

# --- SETUP HALAMAN ---
st.set_page_config(page_title="SEO Super Writer", page_icon="✍️", layout="wide")

# --- AMBIL API KEY DARI SECRETS ---
try:
    api_token = st.secrets["HF_TOKEN"]
except FileNotFoundError:
    st.error("⚠️ API Token belum disetting! Mohon atur di Streamlit Secrets.")
    st.stop()

# --- KONFIGURASI MODEL ---
# Qwen 2.5 7B sangat bagus mengikuti instruksi kompleks ini
model_id = "Qwen/Qwen2.5-7B-Instruct"

# --- FUNGSI GENERATE ---
def get_response(messages):
    client = InferenceClient(model=model_id, token=api_token)
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=2500, # Kita naikkan karena artikel SEO butuh panjang
            temperature=0.6, # Agak rendah agar patuh pada aturan SEO
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# --- UI: INPUT DATA SEO (Sesuai Prompt Anda) ---
with st.sidebar:
    st.header("🛠️ Data SEO")
    target_keyword = st.text_input("Keyword Utama", "Cara Menanam Cabai")
    long_tail = st.text_input("Long Tail Keyword", "Cara menanam cabai rawit di polybag agar berbuah lebat")
    lsi_keywords = st.text_area("LSI Keywords (Pisahkan koma)", "Pupuk organik, Hama tanaman, Media tanam, Panen cabai")
    search_intent = st.selectbox("Search Intent", ["Informational (Edukasi)", "Transactional (Jualan)", "Commercial Investigation (Review)"])
    audiens = st.selectbox("Target Audiens", [
        "Umum / Pemula",
        "Petani & Pekebun Profesional", 
        "Ibu Rumah Tangga / Hobi",
        "Teknisi / Expert"
    ])

st.title("🚀 SEO Article Generator (Senior Editor Mode)")
st.markdown("Tools ini menggunakan prompt **Senior Editor & SEO Specialist** untuk menghasilkan artikel HTML siap posting.")

topic = st.text_input("Judul / Topik Artikel", "Panduan Lengkap Menanam Cabai untuk Pemula")
generate_btn = st.button("✨ Tulis Artikel SEO", type="primary")

# --- LOGIKA UTAMA: PROMPT TEMPLATE ---
if generate_btn:
    st.info("Sedang bertindak sebagai Senior Editor... (Proses bisa memakan waktu 1-2 menit)")
    
    # DISINI KITA MASUKKAN PROMPT RAKSASA ANDA
    # Menggunakan f-string (f""") agar bisa menyisipkan variabel
    
    system_prompt = f"""
    # SYSTEM ROLE & CONTEXT
    Bertindaklah sebagai Senior Editor-in-Chief & SEO Specialist dengan pengalaman lebih dari 20 tahun di media Tier-1.
    Tugas Anda adalah menulis artikel blog format HTML yang mendalam, human-friendly, dan evergreen.

    # TUJUAN UTAMA
    1. Mendominasi Ranking 1 Google.
    2. 100% Terbaca seperti tulisan manusia (Natural, Berempati).
    3. Lolos plagiarisme & AI Detection.
    
    # PANDUAN GAYA PENULISAN (THE HUMAN TOUCH)
    - Gunakan variasi kalimat (pendek vs panjang). Hindari pola monoton.
    - HINDARI kata sambung kaku ("Oleh karena itu", "Selanjutnya"). GANTI dengan ("Nah,", "Masalahnya,", "Asal tahu saja,").
    - Gunakan partikel lokal (lah, kan, kok) secara strategis.
    - Analogi Kehidupan Nyata: Gunakan metafora unik.
    
    # NEGATIVE PROMPTS (DILARANG KERAS)
    - JANGAN gunakan: "Di era digital ini", "Bukan rahasia lagi", "Pada dasarnya", "Kesimpulannya", "Demikianlah ulasan".
    - JANGAN gunakan kata langit: "Komprehensif", "Signifikan", "Masif", "Revolusioner".
    - JANGAN gunakan terjemahan mentah: "Lanskap", "Ranah", "Permadani", "Membuka potensi".
    
    # ATURAN TEKNIS (STRICT HTML OUTPUT)
    - HANYA output kode HTML murni di dalam <body>.
    - Gunakan tag <h2>, <h3>, <p>, <ul>, <ol>.
    - Tabel WAJIB dibungkus <div style="overflow-x: auto;">.
    - Sertakan Image Placeholders setelah H2.
    - JANGAN gunakan Markdown (##, **).
    
    # STRUKTUR KONTEN (SKYSCRAPER)
    1. Meta Title & Description (Hidden comment).
    2. Pendahuluan (Hook emosional + Solusi).
    3. Isi Utama (Golden Keyword Placement, LSI, Snippet Optimization).
    4. Penutup (Call to Action, bukan ringkasan malas).
    
    # INPUT DATA
    1. Topik/Keyword Utama: {target_keyword}
    2. Long Tail Keyword: {long_tail}
    3. Judul Artikel: {topic}
    4. LSI Keywords: {lsi_keywords}
    5. Search Intent: {search_intent}
    6. Target Audiens: {audiens}
    7. Brand Voice: Praktisi Berpengalaman (Jujur, Lugas).

    # EKSEKUSI
    Sekarang, tulislah artikel lengkap berdasarkan Topik dan Data di atas. Patuhi semua aturan HTML dan Gaya Bahasa.
    """

    # Kirim ke AI
    full_response = get_response([{"role": "user", "content": system_prompt}])
    
    if "Error" in full_response:
        st.error(full_response)
    else:
        st.success("✅ Artikel Selesai!")
        
        # Tab untuk melihat hasil
        tab1, tab2 = st.tabs(["📖 Preview (Rendered)", "code HTML Source"])
        
        with tab1:
            # Render HTML agar terlihat seperti di web asli
            st.markdown(full_response, unsafe_allow_html=True)
            
        with tab2:
            st.code(full_response, language='html')
            
        # Tombol Download
        st.download_button("Unduh HTML (.html)", full_response, file_name="artikel_seo.html")
