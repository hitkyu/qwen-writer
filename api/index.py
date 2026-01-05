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
            max_tokens=3000,
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

            # Prompt Template
            system_prompt = f"""
            # ROLE: Senior Editor & SEO Specialist.
            Tulis artikel HTML lengkap (hanya body tag) untuk Blog.
            
            # DATA:
            - Topik: {topic}
            - Keyword: {keyword}
            - Long Tail: {long_tail}
            - LSI: {lsi}
            - Intent: {intent}
            - Target: {audiens}
            
            # ATURAN:
            1. Output HANYA kode HTML (<h2>, <p>, <ul>, <table>).
            2. Gaya bahasa natural, luwes, gunakan partikel 'lah', 'kan'.
            3. Wajib ada Tabel & FAQ.
            4. Jangan pakai markdown, langsung HTML.
            """
            
            # Panggil AI
            result = get_ai_response(system_prompt, api_token)

    return render_template('index.html', result=result, error=error)
