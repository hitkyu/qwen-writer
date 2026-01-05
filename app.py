import streamlit as st
import os
from huggingface_hub import InferenceClient

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="SEO Super Writer Pro",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk tampilan lebih profesional
st.markdown("""
<style>
    .stTextInput > label, .stSelectbox > label, .stTextArea > label {
        font-weight: bold;
        color: #333;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #000;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #333;
        color: white;
    }
    div[data-testid="stForm"] {
        border: 1px solid #ddd;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- 2. API SETUP (VERCEL COMPATIBLE) ---
# Di Vercel, kita mengambil dari Environment Variables, bukan st.secrets lokal jika dideploy
api_token = os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN")

if not api_token:
    st.error("⚠️ API Token tidak ditemukan. Harap atur 'HF_TOKEN' di Environment Variables Vercel atau Streamlit Secrets.")
    st.stop()

# --- 3. LOGIC MODEL ---
model_id = "Qwen/Qwen2.5-7B-Instruct"

def get_response(messages):
    client = InferenceClient(model=model_id, token=api_token)
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=3000, 
            temperature=0.6,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# --- 4. UI LAYOUT (CENTERED & PROFESSIONAL) ---

# Header Section
st.markdown("<div class='main-header'><h1>✍️ SEO Super Writer <span style='color:#FF4B4B'>Pro</span></h1><p>Generate Artikel High-Ranking dengan AI Senior Editor</p></div>", unsafe_allow_html=True)

# Form Container
with st.container():
    with st.form("seo_form"):
        st.subheader("1. Konfigurasi Topik")
        topic = st.text_input("Judul / Topik Artikel", placeholder="Contoh: Panduan Lengkap Menanam Cabai untuk Pemula")
        
        st.subheader("2. Parameter SEO")
        col1, col2 = st.columns(2)
        
        with col1:
            target_keyword = st.text_input("Keyword Utama", placeholder="Contoh: Cara Menanam Cabai")
            search_intent = st.selectbox("Search Intent", ["Informational (Edukasi)", "Transactional (Jualan)", "Commercial Investigation (Review)"])
            
        with col2:
            long_tail = st.text_input("Long Tail Keyword", placeholder="Contoh: Cara menanam cabai di polybag")
            audiens = st.selectbox("Target Audiens", [
                "Umum / Pemula",
                "Petani & Pekebun Profesional", 
                "Ibu Rumah Tangga / Hobi",
                "Teknisi / Expert",
                "Mahasiswa / Akademisi"
            ])
        
        st.markdown("---")
        st.subheader("3. Konteks Tambahan (LSI)")
        lsi_keywords = st.text_area("LSI Keywords (Pisahkan dengan koma)", placeholder="Pupuk organik, Hama tanaman, Media tanam, Panen cabai", help="Keyword pendukung agar artikel lebih relevan di mata Google.")
        
        # Tombol Submit (Full Width karena CSS di atas)
        submitted = st.form_submit_button("✨ Generate Artikel Sekarang")

# --- 5. EXECUTION & OUTPUT ---
if submitted:
    if not topic or not target_keyword:
        st.warning("⚠️ Harap isi setidaknya Judul Topik dan Keyword Utama.")
    else:
        with st.status("🤖 Sedang bekerja...", expanded=True) as status:
            st.write("🔍 Menganalisis Search Intent & Audiens...")
            st.write("🧠 Merancang struktur artikel Skyscraper...")
            st.write("✍️ Menulis konten dengan gaya human-touch...")
            
            # --- PROMPT TEMPLATE ---
            system_prompt = f"""
            # SYSTEM ROLE
            Bertindaklah sebagai Senior Editor-in-Chief & SEO Specialist (20+ tahun pengalaman).
            Tugas: Tulis artikel blog format HTML yang mendalam, human-friendly, dan evergreen.

            # DATA INPUT
            - Topik: {topic}
            - Keyword Utama: {target_keyword}
            - Long Tail: {long_tail}
            - LSI Keywords: {lsi_keywords}
            - Intent: {search_intent}
            - Audiens: {audiens}

            # GUIDELINES (STRICT)
            1. **Format:** HTML Murni (Hanya body content: <h2>, <h3>, <p>, <ul>, <ol>, <table>).
            2. **Style:** Natural, Berempati, Variatif (Pendek/Panjang), Gunakan partikel lokal (lah, kan, kok).
            3. **Forbidden:** JANGAN gunakan frasa AI klise ("Di era digital", "Kesimpulannya", "Signifikan").
            4. **Structure:** - Hook Emosional (Pendahuluan)
               - Isi Utama (Deep Dive + Tabel jika perlu)
               - FAQ Section (Wajib ada)
               - Penutup (Call to Action kuat)

            # EKSEKUSI
            Tulis artikel lengkap sekarang. Pastikan tabel dibungkus <div style="overflow-x: auto;">.
            """

            full_response = get_response([{"role": "user", "content": system_prompt}])
            
            status.update(label="✅ Selesai!", state="complete", expanded=False)

        # Result Display
        if "Error" in full_response:
            st.error(full_response)
        else:
            st.divider()
            st.subheader("📄 Hasil Artikel")
            
            tab1, tab2 = st.tabs(["📖 Preview Tampilan", "code HTML Source"])
            
            with tab1:
                st.markdown(full_response, unsafe_allow_html=True)
            
            with tab2:
                st.code(full_response, language='html')
                
            st.download_button(
                label="📥 Download File HTML",
                data=full_response,
                file_name=f"{target_keyword.replace(' ', '_')}.html",
                mime="text/html"
            )
