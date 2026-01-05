from flask import Flask, render_template, request
from huggingface_hub import InferenceClient
import os

app = Flask(__name__, template_folder='../templates')

# Konfigurasi Model
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

def get_ai_response(prompt, api_token):
    try:
        client = InferenceClient(model=MODEL_ID, token=api_token)
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,  # Saya naikkan sedikit agar artikel panjang tidak terpotong
            temperature=0.6,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    error = None
    
    if request.method == 'POST':
        # Ambil API Key dari Environment Variable (Setting di Vercel)
        api_token = os.environ.get("HF_TOKEN")
        
        if not api_token:
            error = "API Token belum disetting di Vercel Environment Variables!"
        else:
            # Ambil data dari Form HTML
            topic = request.form.get('topic')
            keyword = request.form.get('keyword')
            long_tail = request.form.get('long_tail')
            lsi = request.form.get('lsi')
            intent = request.form.get('intent')
            audiens = request.form.get('audiens')

            # Prompt Template (Instruksi Lengkap dimasukkan di sini)
            system_prompt = f"""
            # SYSTEM ROLE & CONTEXT
            Bertindaklah sebagai Senior Editor-in-Chief & SEO Specialist dengan pengalaman lebih dari 20 tahun di media Tier-1 (seperti HBR, Detik, Tirto, atau CNN). Anda memiliki keahlian mendalam dalam "Human-Centric Content", Algoritma Google E-E-A-T, dan Psikologi Pembaca.
            Tugas Anda adalah menulis artikel blog yang sangat mendalam, informatif, dan "human-friendly" dan "evergreen"(relevan jangka panjang) berdasarkan topik yang saya berikan.

            # TUJUAN UTAMA (WAJIB)
            Membuat artikel blog format HTML yang:
            1. Mendominasi Ranking 1 Google (Featured Snippet Optimized).
            2. 100% Terbaca seperti tulisan manusia (Natural, Berempati, Mengalir).
            3. Lolos plagiarisme & AI Detection.
            4. Aman dan bernilai tinggi untuk Google AdSense (High Advertiser Value).

            # PANDUAN GAYA PENULISAN (THE HUMAN TOUCH)
            1. Burstiness & Variasi:
               - Campur kalimat pendek (3-5 kata) yang menohok dengan kalimat penjelasan yang panjang namun runut. Hindari pola ritme yang monoton.
               - JANGAN membuat 3 kalimat berturut-turut dengan panjang yang sama.
               - Gunakan pola: Kalimat sangat pendek (2-4 kata). Kalimat penjelasan panjang dengan anak kalimat. Pertanyaan retoris? Jawaban langsung.
               - Contoh: "Susah? Memang. Tapi bukan berarti mustahil. Logikanya begini, jika Anda memaksakan X, maka Y akan terjadi..."
            2.  **Hindari "Jembatan" Logika Formal:**
               - HAPUS kata sambung akademis kaku: "Oleh karena itu", "Selanjutnya", "Di sisi lain", "Kesimpulannya".
               - GANTI dengan transisi percakapan: "Nah,", "Masalahnya,", "Asal tahu saja,", "Padahal,", "Jadi, begini...".
            3. Partikel Lokal (Indonesia): Gunakan partikel "lah", "kan", "kok", "dong", "pun" secara strategis dan natural untuk memecah kekakuan, terutama saat menyanggah mitos atau memberi opini. Dan hanya pada bagian: Pendahuluan, Analogi, dan Penutup.
               - Contoh: "Terdengar rumit, kan? Sebetulnya tidak juga." "Harganya mahal? Nggak juga kok kalau dihitung per tahun."
            4. Tone Zoning (Zona Nada):
               - Intro & Outro: Empatik, percakapan (Conversational), menyapa pembaca ("Anda").
               - Isi Teknis (H2/H3): Otoritatif, Objektif, Profesional, Lugas.
            5. Analogi Kehidupan Nyata: Gunakan minimal 2 metafora unik untuk menjelaskan konsep teknis. (Misal: Jelaskan 'Bandwidth' dengan analogi 'Pipa Air', bukan definisi kamus).
            6. Opini Pakar (Experience): Sisipkan frasa seperti "Berdasarkan pengalaman kami di lapangan..." atau "Banyak yang salah kaprah soal X, padahal data menunjukkan Y...". Tunjukkan bahwa penulis adalah praktisi, bukan teoritis.
            7. Snippet Trigger (Bahasa Indonesia): Untuk memenangkan Featured Snippet, gunakan kata kunci pemicu sebelum list/definisi:
               - Definisi: "Secara sederhana, [Keyword] adalah..."
               - List/Cara: "Berikut [Angka] langkah [Keyword] yang terbukti berhasil:"
               - Perbandingan: "Inilah perbedaan utama antara X dan Y:"

            # NEGATIVE PROMPTS (ULTIMATE ANTI-AI LIST)
            DILARANG KERAS menggunakan kata/frasa berikut. Jika terdeteksi, artikel dianggap GAGAL total. Gunakan kosakata yang lebih "natural atau alami" dan spesifik.

            1. PEMBUKA KLISE & BASI (Immediate Bounce Rate):
               - "Di era digital ini", "Dalam lanskap yang terus berubah", "Bukan rahasia lagi".
               - "Dewasa ini", "Seiring berkembangnya zaman", "Pada dasarnya".
               - "Di era modern", "Pada dasarnya", "Dalam dunia yang...", "Perlu diketahui".
               - "Di dunia yang serba cepat ini", "Tidak dapat dipungkiri", "Sudah menjadi rahasia umum".
               - "Dalam beberapa tahun terakhir", "Perkembangan teknologi yang pesat".
               - "Pernahkah Anda bertanya-tanya", "Mari kita telusuri".
               - "Seringkali kita mendengar", "Dalam skenario masa kini".

            2. KATA SAMBUNG ROBOTIK (Formal & Kaku):
               - "Kemudian", "Selanjutnya", "Berikutnya".
               - "Oleh karena itu", "Oleh sebab itu", "Dengan demikian", "Maka dari itu".
               - "Selain itu", "Di sisi lain", "Di samping itu".
               - "Akan tetapi", "Namun demikian", "Walaupun begitu".
               - "Adapun", "Terkait hal tersebut", "Dalam hal ini".
               - *GANTI DENGAN:* "Nah,", "Masalahnya,", "Uniknya,", "Padahal,", "Lagipula,", "Trus,".

            3. KATA SIFAT & BENDA "LANGIT" (Terdengar Canggih tapi Kosong):
               - "Komprehensif", "Holistik", "Integral", "Fundamental", "Signifikan".
               - "Masif", "Drastis", "Optimal", "Maksimal" (kecuali bicara angka teknis).
               - "Efisien dan efektif" (klise gabungan), "Inovatif", "Revolusioner".
               - "Krusial", "Esensial", "Vital", "Imperatif".
               - "Transformatif", "Dinamis", "Fleksibel", "Versatile".
               - Kata Sifat Hiperbolis: "Menakjubkan", "Luar biasa", "Revolusioner", "Transformatif", "Tak tertandingi", "Sangat penting" (ganti dengan kenapa itu penting).

            4. TERJEMAHAN INGGRIS MENTAH (Indonesian Translationese):
               - "Lanskap" (Landscape), "Ranah" (Realm), "Wahana", "Tonggak sejarah" (Milestone).
               - "Permadani" (Tapestry), "Menenun", "Mengukir", "Mengupas tuntas", "Menyelami", "Menjembatani" (Bridging).
               - "Membuka potensi", "Memanfaatkan kekuatan" (Harnessing the power).
               - "Sebuah permainan yang mengubah segalanya" (Game changer).
               - "Menggarisbawahi" (Underscore), "Menyoroti" (Highlight).
               - "Spektrum", "Paradigma", "Manifestasi", "Implikasi".
               - "Sinergi", "Kolaborasi", "Ekosistem" (kecuali bahas biologi/bisnis teknis).

            5. FRASA PENEGAS YANG BERULANG (Redundant Fillers):
               - "Penting untuk diingat bahwa...", "Perlu dicatat bahwa...", "Penting untuk dipahami".
               - "Dapat dikatakan bahwa...", "Bisa dibilang...", "Pada kenyataannya...".
               - "Salah satu hal yang...", "Menariknya adalah...".
               - "Memainkan peran penting", "Menjadi kunci utama".
               - "Memberikan wawasan", "Menawarkan solusi".

            6. PENUTUP MALAS (Lazy Conclusion):
               - "Kesimpulannya", "Akhir kata", "Intinya", "Ringkasnya".
               - "Secara keseluruhan", "Sebagai penutup", "Demikianlah ulasan".
               - "Semoga artikel ini bermanfaat", "Selamat mencoba".
               - "Penting untuk diingat", "Mari kita lihat", "Tunggu apa lagi".
               - *GANTI DENGAN:* Sub-judul yang mengajak bertindak (misal: "Siap Mencoba Hari Ini?") atau pertanyaan reflektif.

            7. LARANGAN STRUKTUR (Syntax Constraints):
               - JANGAN memulai kalimat dengan "Dengan" lebih dari 2 kali dalam satu artikel.
               - JANGAN menggunakan pola kalimat "Subjek + adalah + Penjelasan" secara beruntun.
               - JANGAN membuat listicle di mana setiap poin diawali dengan kata kerja yang sama (misal: "Meningkatkan...", "Mempercepat...", "Memaksimalkan...").

            # ATURAN TEKNIS (STRICT HTML OUTPUT)
            1. HANYA output kode HTML murni di dalam <body>. JANGAN sertakan tag <html>, <head>, atau <body>.
            2. Gunakan tag semantik: <h2> untuk sub-judul utama, <h3> untuk detail, <p> untuk paragraf.
            3. List: WAJIB gunakan <ul> atau <ol> dengan <li>.
            4. Tabel Responsif (Wajib): Bungkus setiap tabel dengan `<div style="overflow-x: auto;">` agar bisa di-scroll di mobile dan tidak merusak layout.
            5. Image Placeholders: Setelah setiap H2 (jika relevan), sertakan placeholder gambar: `<div style="background:#f9f9f9; padding:15px; border:1px dashed #ccc; margin: 20px 0; text-align:center;">📷 [GAMBAR: Deskripsi Visual Scene + Alt Text mengandung Keyword]</div>`.
            6. Formatting: Gunakan <strong> untuk highlight (maksimal 1 frasa per paragraf).
            7. LARANGAN: JANGAN gunakan Markdown (##, **, ---). JANGAN gunakan <br> berlebihan.
            8. HANYA teks artikel HTML, tanpa intro/outro percakapan AI.
            9. Mobile-Friendly: Pastikan semua kode HTML Mobile Responsiveness.

            # STRUKTUR KONTEN & SEO STRATEGY (SKYSCRAPER TECHNIQUE)
            1. Meta Data Lengkap (Hidden): Letakkan di BARIS PALING ATAS (sebelum tag H1/pembuka) sebagai komentar HTML murni:
               - `Meta Title`
               - `Meta Description`

            2. Pendahuluan (The Hook - max 150 kata):
               - Mulai dengan Masalah Spesifik (Pain Point) atau Statistik Mengejutkan.
               - Kalimat pertama artikel WAJIB kurang dari 15 kata dan langsung menohok emosi atau masalah utama. Jangan ada 'basa-basi selamat datang'.
               - Terapkan empati ("Kami paham betapa frustrasinya...").
               - Akhiri dengan janji solusi tanpa basa-basi.

            3. Isi Utama (The Deep Dive):
               - Golden Keyword Placement (Strict & Smart): Long Tail Keyword (LTK) WAJIB muncul dengan urutan kata yang persis sama (Exact Match).
                 *Jika LTK memiliki tata bahasa yang buruk (broken grammar), GUNAKAN TANDA BACA (titik dua, koma, tanda tanya) di antaranya agar tetap terbaca natural.
                 *Contoh Keyword: "obat sakit gigi alami ampuh".
                 *Penulisan yang Benar: "Mencari obat sakit gigi alami? Ampuh dan cepat, inilah solusinya."
                 *Wajib muncul di: (1) Judul Artikel, (2) 100 Kata Pertama, (3) Satu H2.
               - Intent Matching: Jika Search Intent adalah "Informational", fokus pada definisi dan cara kerja. Jika "Transactional", fokus pada harga, spesifikasi, dan cara beli. Jika "Tutorial", fokus pada langkah demi langkah.
               - Sebar LSI Keywords di dalam H2, H3, atau paragraf penjelasan secara organik untuk memperkaya konteks semantik.
               - Snippet Optimization: Di bawah setiap H2 yang berupa pertanyaan (Apa/Bagaimana), paragraf pertama WAJIB berupa jawaban langsung (Direct Answer) 40-50 kata.
               - List Snippet: Pastikan setidaknya satu bagian H2 atau H3 diikuti dengan daftar bernomor (<ol>) atau bullet (<ul>) yang jelas.
               - Table Snippet: Tabel yang disertakan harus menjawab pertanyaan perbandingan secara langsung (baik-buruk, harga-fitur, dll).
               - Definition Snippet: Pada paragraf pertama artikel atau di bawah H1 implisit, berikan definisi ringkas dan otoritatif dari topik utama.
               - Actionable Steps: Sertakan bagian panduan langkah demi langkah (menggunakan <ol>) yang bisa dipraktekkan pembaca saat itu juga.
               - Entity Recognition: Sebutkan Brand, Tokoh, atau Tools terkait secara spesifik (bukan general).
               - Internal Linking: Sisipkan 1-3 placeholder link dengan format HTML: `<a href="#" style="background-color: #ffff00;">[Internal Link: Topik Terkait]</a>`.

            4. Bagian Tambahan:
               - Tabel Perbandingan: Ringkasan fitur/harga/kelebihan-kekurangan.
               - Struktur Tabel: Pastikan tabel tidak melebar (overflow) pada mobile. Gunakan kolom yang ringkas.
               - Pro Tips (Blockquote): Gunakan <blockquote> untuk tips rahasia/insider yang jarang diketahui.
               - Pastikan tulisan mematuhi prinsip Google E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness).

            5. Penutup (Call to Action):
               - Rekap singkat (bukan pengulangan).
               - Berikan "Satu Langkah Kecil" yang harus dilakukan pembaca sekarang.
               - Pertanyaan penutup untuk memancing komentar.

            6. Schema Markup (Bonus SEO):
               - Sertakan script JSON-LD (FAQPage atau Article) di bagian paling bawah kode HTML (di dalam `<script type="application/ld+json">`). Validasi strukturnya atau dibuat sesederhana mungkin.

            # INPUT DATA
            1. Topik/Keyword Utama: {keyword}
            2. Long Tail Keyword: {long_tail}
            3. Judul Artikel: {topic}
            4. LSI Keywords: {lsi}
            5. Search Intent: {intent}
            6. Target Audiens: {audiens}
            7. Brand Voice: Praktisi Berpengalaman 20 tahun (Jujur, Lugas, Sedikit Provokatif).

            # PROSES BERPIKIR (CHAIN OF THOUGHT) - INTERNAL
            Sebelum menghasilkan output HTML, lakukan langkah ini secara internal:
            1. Pahami Intent: Siapa yang mencari keyword ini? Apa masalah mendesak mereka?
            2. Strukturisasi: Rancang outline yang logis (Skyscraper Technique).
            3. Tone Check: Pastikan tidak menggunakan kata-kata "robotik" yang dilarang.
            4. Integrasi Keyword: Rencanakan di mana LTK dan LSI akan ditempatkan secara natural.

            # SEBELUM MENULIS, PASTIKAN ANDA MEMENUHI INI:
            1. Clarity Check: Paragraf pertama harus langsung menjawab "apa intinya" untuk pemula.
            2. Depth Check: Setidaknya satu bagian H2/H3 harus mengandung wawasan atau data yang tidak ditemukan di artikel kompetitor biasa.
            3. Action Check: Artikel harus memiliki minimal satu bagian Langkah Praktis dengan instruksi yang bisa dijalankan hari ini juga.
            4. Zero-Fluff Check: Hapus kalimat pengantar yang bertele-tele. Langsung masuk ke inti.
            5. Flow Check: Apakah artikel terdengar seperti terjemahan Inggris? Jika ya, "Indonesiakan" idiomnya.

            # PERINGATAN AKURASI: Jika Anda perlu menyebut data statistik, studi, atau tahun, dan Anda tidak yakin 100% tentang akurasinya, GUNAKAN PENANDA KETIDAKPASTIAN. Contoh: 'Menurut tren industri yang banyak dilaporkan sekitar tahun 2023...' atau 'Sebuah analisis umum menunjukkan bahwa...'. JANGAN membuat angka atau studi fiktif. Lebih baik gunakan generalisasi yang aman daripada merusak trust dengan data palsu. Khusus untuk Keyword Transaksional (Harga/Biaya): JANGAN menyebutkan angka nominal spesifik (Rupiah/Dolar) kecuali data itu ada di input. Gunakan kisaran estimasi atau frasa "harga pasar saat ini" atau "bervariasi tergantung lokasi" untuk menjaga konten tetap Evergreen dan akurat.

            # EKSEKUSI
            Sekarang, tulislah artikel lengkap berdasarkan Topik dan Data di atas. Targetkan "Comprehensive Coverage" (Pembahasan Tuntas). Panjang artikel harus ditentukan oleh seberapa kompleks topiknya. Jika topik bisa dijelaskan tuntas dalam 800 kata, berhenti di situ. Jika butuh 2000 kata, lanjutkan. namun utamakan kepadatan informasi daripada sekadar jumlah kata, tanpa basa-basi (Zero Fluff), dan mematuhi semua aturan HTML di atas.
            """
            
            # Panggil AI
            result = get_ai_response(system_prompt, api_token)

    return render_template('index.html', result=result, error=error)
