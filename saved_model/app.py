# app.py
import streamlit as st
import torch
import re
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Fungsi pembersihan teks berdasarkan pilihan pengguna
def cleaning(text, remove_numbers=False, remove_urls=False, remove_emojis=False, remove_extra_spaces=False, lowercase=False):
    if remove_urls:
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Hapus URL
    if remove_numbers:
        text = re.sub(r'\d+', '', text)  # Hapus angka
    if remove_emojis:
        text = re.sub(r'[^\w\s]', '', text)  # Hapus emoji dan tanda baca
    if remove_extra_spaces:
        text = ' '.join(text.split())  # Hapus spasi berlebih
    if lowercase:
        text = text.lower()  # Konversi ke huruf kecil
    return text.strip()

# Fungsi untuk mendapatkan alasan berdasarkan attention weights
def get_explanation(review, model, tokenizer, device, predicted_label, preproc_options):
    try:
        cleaned_review = cleaning(
            review,
            remove_numbers=preproc_options['remove_numbers'],
            remove_urls=preproc_options['remove_urls'],
            remove_emojis=preproc_options['remove_emojis'],
            remove_extra_spaces=preproc_options['remove_extra_spaces'],
            lowercase=preproc_options['lowercase']
        )
        encodings = tokenizer(
            [cleaned_review],
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt',
            return_attention_mask=True
        )
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            attentions = outputs.attentions
            last_layer_attention = attentions[-1][0, 0].cpu().numpy()
            mean_attention = last_layer_attention.mean(axis=0)

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        token_scores = list(zip(tokens, mean_attention))
        important_tokens = [
            (token, score) for token, score in token_scores
            if token not in ['[CLS]', '[SEP]', '[PAD]'] and not token.startswith('##')
        ]
        important_tokens = sorted(important_tokens, key=lambda x: x[1], reverse=True)[:3]
        reasons = [token for token, _ in important_tokens]

        label_map = {0: 'CG (Palsu)', 1: 'OR (Asli)'}
        explanation = f"Ulasan diklasifikasikan sebagai {label_map[predicted_label]} karena kata-kata seperti {', '.join(reasons)} memiliki pengaruh besar pada prediksi."
        return explanation
    except Exception as e:
        return f"Gagal menghasilkan alasan: {e}"

# Fungsi prediksi
def predict_review(review, model, tokenizer, device, preproc_options):
    try:
        cleaned_review = cleaning(
            review,
            remove_numbers=preproc_options['remove_numbers'],
            remove_urls=preproc_options['remove_urls'],
            remove_emojis=preproc_options['remove_emojis'],
            remove_extra_spaces=preproc_options['remove_extra_spaces'],
            lowercase=preproc_options['lowercase']
        )
        encodings = tokenizer(
            [cleaned_review],
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        encodings = {key: val.to(device) for key, val in encodings.items()}
        model.eval()
        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_label = np.argmax(probs)
        label_map = {0: 'CG (Palsu)', 1: 'OR (Asli)'}
        predicted_label = label_map[pred_label]
        prob_or = probs[1] * 100
        return predicted_label, prob_or, cleaned_review, pred_label, review
    except Exception as e:
        return None, None, f"Error saat memproses ulasan: {e}", None, review

# Konfigurasi Streamlit
st.set_page_config(page_title="Deteksi Ulasan Palsu", layout="centered")

# Sidebar untuk pilihan pre-processing
with st.sidebar:
    st.header("Pilih Tahap Pre-processing")
    st.markdown("Pilih langkah pre-processing untuk membersihkan ulasan sebelum diproses model:")
    
    remove_numbers = st.checkbox("Hapus Angka", value=False, help="Menghapus semua angka (contoh: '123' dihapus).")
    remove_urls = st.checkbox("Hapus URL", value=True, help="Menghapus tautan seperti http://example.com.")
    remove_emojis = st.checkbox("Hapus Emoji/Tanda Baca", value=False, help="Menghapus emoji dan tanda baca (contoh: 😊, !).")
    remove_extra_spaces = st.checkbox("Hapus Spasi Berlebih", value=True, help="Menghapus spasi ganda atau berlebih.")
    lowercase = st.checkbox("Konversi ke Huruf Kecil", value=False, help="Mengubah semua teks menjadi huruf kecil.")
    
    preproc_options = {
        'remove_numbers': remove_numbers,
        'remove_urls': remove_urls,
        'remove_emojis': remove_emojis,
        'remove_extra_spaces': remove_extra_spaces,
        'lowercase': lowercase
    }
    
    st.markdown("""
    **Tujuan Pre-processing**: Membersihkan teks agar model fokus pada konten relevan, meningkatkan akurasi prediksi.
    **Catatan**: Tokenisasi (mengubah teks menjadi token untuk model) dilakukan otomatis setelah pembersihan.
    """)

# Halaman utama
st.title("Deteksi Ulasan Palsu Bahasa Indonesia")
st.write("Masukkan ulasan untuk mengetahui apakah termasuk ulasan asli (OR) atau ulasan palsu (CG)")

# Memuat model dan tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    try:
        model_path = 'C:/Users/ACER/Downloads/saved_model'
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        device = torch.device('cpu')
        model.to(device)
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Gagal memuat model atau tokenizer: {e}")
        return None, None, None

model, tokenizer, device = load_model_and_tokenizer()
if model is None or tokenizer is None:
    st.stop()

# Input ulasan
st.subheader("Masukkan Ulasan")
user_input = st.text_area("Ketik ulasan:", height=150, placeholder="Contoh: Produk ini sangat bagus 123, cek di https://example.com! 😊")

# Tombol prediksi
if st.button("Deteksi Ulasan"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan ulasan!")
    else:
        with st.spinner("Memproses..."):
            predicted_label, prob_or, cleaned_review, pred_label, original_review = predict_review(
                user_input, model, tokenizer, device, preproc_options
            )
            if predicted_label is None:
                st.error(cleaned_review)
            else:
                st.subheader("Hasil Deteksi")
                
                # Perbandingan pre-processing
                st.subheader("Perbandingan Pre-processing")
                comparison_df = pd.DataFrame({
                    "Sebelum": [original_review],
                    "Sesudah": [cleaned_review]
                })
                st.table(comparison_df)
                
                # Hasil prediksi
                st.write(f"**Prediksi**: {predicted_label}")
                st.write(f"**Probabilitas Asli (OR)**: {prob_or:.2f}%")
                
                # Alasan klasifikasi
                st.subheader("Alasan Klasifikasi")
                explanation = get_explanation(user_input, model, tokenizer, device, pred_label, preproc_options)
                st.write(explanation)
                
                # Visualisasi probabilitas
                st.subheader("Distribusi Probabilitas")
                probs_df = pd.DataFrame({
                    'Kelas': ['CG (Palsu)', 'OR (Asli)'],
                    'Probabilitas (%)': [100 - prob_or, prob_or]
                })
                st.bar_chart(probs_df.set_index('Kelas')['Probabilitas (%)'])

st.markdown("---")
st.markdown("Dibuat dengan Streamlit dan DistilBERT.")