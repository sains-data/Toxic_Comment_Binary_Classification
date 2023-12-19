import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import re
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import gdown
import os
# URL dari "Bagikan" tautan Google Drive
url = 'https://drive.google.com/uc?id=1iTxhqpA8rHhCelQxu9DAN20NzNwd49nj'
url_pickle = 'https://drive.google.com/uc?id=18DN21jgVVaEdov5liEG_ez49s8PoxyaN'

# Nama file output tempat file model akan disimpan setelah diunduh
output = 'modelku.h5'
output_pickle = 'tokenizer.pickle'

# Mengunduh file dari Google Drive
gdown.download(url, output, quiet=False)
gdown.download(url_pickle, output_pickle, quiet=False)

# Memeriksa apakah file telah diunduh dengan sukses dan tersedia di direktori tersebut
if os.path.exists(output):
    # Memuat model
    model = load_model(output)
else:
    raise Exception("File model tidak ditemukan setelah unduhan.")

# Memuat model dan tokenizer
if os.path.exists(output):
    # Memuat tokenizer dari file pickle
    with open(output_pickle,'rb') as handle:
        x_tokenizer = pickle.load(handle)
else:
    st.error("File tokenizer tidak ditemukan setelah unduhan.")


max_text_length = 400  # sesuaikan dengan settingan Anda

# Fungsi pembersihan teks
def clean_punctuation(text):
    text = str(text)
    # Case folding
    text = text.lower()
    # Menghapus spasi berlebih
    text = ' '.join(text.split())
    # substitusi kata
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"r", "", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    # Menghapus karakter khusus, tanda baca
    text = re.sub(r'[-.,+"&\'#@;:{}`+=~/!?()]', '', text)\
    # memastikan tanda baca terhapus
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub("(\W)"," ",text)
    # menghapus kata yang diapit oleh karakter s
    text = re.sub('S*dS*s*','', text)           

    
    return text

# UI Streamlit
st.title("Aplikasi Prediksi Komentar Beracun")
teks_komentar = st.text_area("Masukkan Komentar:")
file_excel = st.file_uploader("Atau unggah file dengan komentar (Excel atau CSV):", type=["xlsx", "csv"])

if st.button("Prediksi"):
    if teks_komentar:
        # Proses teks dan lakukan prediksi
        cleaned_text = clean_punctuation(teks_komentar)
        tokenized_text = x_tokenizer.texts_to_sequences([cleaned_text])
        padded_text = pad_sequences(tokenized_text, maxlen=max_text_length)
        # Karena model memiliki dua input, kita perlu memberikan input yang sama ke kedua channel
        prediksi = model.predict([padded_text, padded_text])
        st.write("Hasil Prediksi:", "Beracun" if prediksi[0][0] > 0.5 else "Tidak Beracun")

    if file_excel is not None:
        df = pd.read_excel(file_excel)
        df['cleaned'] = df['kolom_komentar'].apply(clean_punctuation)
        tokenized_df = x_tokenizer.texts_to_sequences(df['cleaned'].values)
        padded_df = pad_sequences(tokenized_df, maxlen=max_text_length)
        prediksi_df = model.predict([padded_df, padded_df])
        df['Prediksi'] = np.where(prediksi_df.flatten() > 0.5, "Beracun", "Tidak Beracun")
        st.write(df[['kolom_komentar', 'Prediksi']])
        
        # Ambil komentar yang dianggap beracun dan tidak beracun
        komentar_beracun = ' '.join(df[df['Prediksi'] == 'Beracun']['cleaned'].tolist())
        komentar_tidak_beracun = ' '.join(df[df['Prediksi'] == 'Tidak Beracun']['cleaned'].tolist())
        
        # Hanya buat word cloud jika ada kata-kata dalam teks
        if komentar_beracun:
            wordcloud_beracun = WordCloud(width=800, height=400, background_color ='white', max_words=200, contour_width=3, contour_color='firebrick').generate(komentar_beracun)
            st.subheader('Word Cloud untuk Komentar Beracun')
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud_beracun, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
            
        if komentar_tidak_beracun:
            wordcloud_tidak_beracun = WordCloud(width=800, height=400, background_color ='white', max_words=200, contour_width=3, contour_color='steelblue').generate(komentar_tidak_beracun)
            st.subheader('Word Cloud untuk Komentar Tidak Beracun')
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud_tidak_beracun, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
    
