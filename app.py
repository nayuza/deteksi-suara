import streamlit as st
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D, concatenate,
    TimeDistributed, Flatten, Bidirectional, GRU, Dense
)
from tensorflow.keras.utils import load_img, img_to_array
import os
import tempfile
import logging

# --- KONFIGURASI APLIKASI ---
# DIMODIFIKASI: Menambahkan konfigurasi halaman untuk tampilan yang lebih baik
st.set_page_config(
    page_title="Deteksi Suara",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="auto"
)

# Menonaktifkan pesan log yang tidak perlu dari TensorFlow
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- FUNGSI UNTUK MEMBANGUN ARSITEKTUR MODEL ---
def FLB(inp):
    x = Conv2D(filters=120, kernel_size=(9, 9), strides=(2, 2), activation='relu', padding='same')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=384, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    return x

def time_attention(inp):
    x = Conv2D(filters=64, kernel_size=(1, 9), activation='relu', padding='same')(inp)
    x = Conv2D(filters=64, kernel_size=(1, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters=64, kernel_size=(1, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    return x

def frequency_attention(inp):
    x = Conv2D(filters=64, kernel_size=(9, 1), activation='relu', padding='same')(inp)
    x = Conv2D(filters=64, kernel_size=(3, 1), activation='relu', padding='same')(x)
    x = Conv2D(filters=64, kernel_size=(3, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    return x

def MAM(inp):
    ta = time_attention(inp)
    fa = frequency_attention(inp)
    mam = concatenate([ta, fa])
    mam = BatchNormalization()(mam)
    return mam

def create_model_architecture(input_shape=(64, 64, 3), num_classes=12):
    """Membangun arsitektur model CRNN yang sama persis dengan yang dilatih."""
    inp = Input(shape=input_shape)
    x = FLB(inp)
    mam = MAM(x)
    x = concatenate([x, mam])
    x = FLB(x)
    x = TimeDistributed(Flatten())(x)
    x = Bidirectional(GRU(128, return_sequences=True, dropout=0.3))(x)
    x = Bidirectional(GRU(128, return_sequences=False, dropout=0.3))(x)
    x = Dense(80, activation='relu')(x)
    x = BatchNormalization()(x)
    out = Dense(units=num_classes, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    return model

# --- KELAS DETEKTOR ---
class AgeGenderDetector:
    def __init__(self, model_path):
        self.model = create_model_architecture()
        self.model.load_weights(model_path)
        self._initializeLabelDecoder()
        self.sampling_rate = 16000
        self.n_fft = 256
        self.num_overlap = 128
        self._spec_img_size = (64, 64)

    def _initializeLabelDecoder(self):
        self._decode_gender_age = {0:'Perempuan - Remaja', 1:'Perempuan - 20an', 2:'Perempuan - 30an', 3:'Perempuan - 40an', 4:'Perempuan - 50an', 5:'Perempuan - 60an',
                                 6:'Laki-laki - Remaja', 7:'Laki-laki - 20an', 8:'Laki-laki - 30an', 9:'Laki-laki - 40an', 10:'Laki-laki - 50an', 11:'Laki-laki - 60an'}

    def scale_minmax(self, X, min=0, max=255):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled

    def saveSpectrogram(self, data, fn):
        plt.figure(figsize=(1, 1), dpi=self._spec_img_size[0])
        plt.axis('off')
        plt.imshow(data, aspect='auto', origin='lower', interpolation='none')
        plt.tight_layout(pad=0)
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)
        plt.close()

    def _extractSpectrogram(self, audio_file):
        y, sr = librosa.load(audio_file, sr=self.sampling_rate)
        spec = librosa.stft(y, n_fft=self.n_fft, hop_length=self.num_overlap)
        spec = librosa.amplitude_to_db(np.abs(spec))
        img = self.scale_minmax(spec).astype(np.uint8)
        out_file = 'temp_spec.png'
        self.saveSpectrogram(img, out_file)
        inp_spec = img_to_array(load_img(out_file, target_size=self._spec_img_size))
        return inp_spec
    
    def getPredictionLabelName(self, encod_lab):
        return self._decode_gender_age.get(encod_lab)

    def get_top_predictions(self, audio_file, top_k=5):
        inp_spec = self._extractSpectrogram(audio_file)
        inp_spec = np.expand_dims(inp_spec/255.0, axis=0)
        inp_spec = inp_spec.reshape(-1, 64, 64, 3)
        prediction_probs = self.model.predict(inp_spec, verbose=0)[0]
        top_indices = np.argsort(prediction_probs)[-top_k:][::-1]
        results = []
        for i in top_indices:
            label = self.getPredictionLabelName(i)
            probability = prediction_probs[i]
            results.append((label, probability))
        return results

# --- ANTARMUKA STREAMLIT ---
@st.cache_resource
def load_app_model():
    """Memuat AgeGenderDetector hanya sekali."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'best_model_gender_age.h5')
        
        if not os.path.exists(model_path):
            st.error(f"File model tidak ditemukan di path yang diharapkan: {model_path}")
            return None
        
        detector = AgeGenderDetector(model_path)
        return detector
    except Exception as e:
        st.error(f"Terjadi error saat memuat model: {e}")
        return None

def main():
    # --- UI DIMODIFIKASI ---
    st.title("🎙️Deteksi Usia & Gender Suara")
    st.markdown("Unggah file audio untuk dianalisis oleh model **CNN-RNN**. Aplikasi akan memprediksi rentang usia dan gender berdasarkan karakteristik suara.")
    st.markdown("---")

    detector = load_app_model()
    
    if detector is None:
        st.error("Aplikasi tidak dapat berjalan karena model gagal dimuat. Harap periksa kembali file model Anda.")
        return

    # Kontainer untuk area unggah
    with st.container(border=True):
        # DIMODIFIKASI: Menghilangkan "Langkah" dan membuat tulisan ke tengah
        st.markdown("<h3 style='text-align: center; color: #1E88E5;'>Unggah File Suara Anda</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Pilih file audio (.mp3 atau .wav)",
            type=["mp3", "wav"],
            label_visibility="collapsed"
        )

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        # Tombol deteksi dibuat lebih menonjol
        if st.button("🕵 Mulai Deteksi Sekarang!", type="primary", use_container_width=True):
            with st.spinner('Sedang menganalisis suara... Harap tunggu...'):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_audio_path = tmp_file.name
                
                try:
                    top_predictions = detector.get_top_predictions(temp_audio_path, top_k=5)
                    main_prediction_label = top_predictions[0][0]
                    main_prediction_prob = top_predictions[0][1]

                    st.success("✅ Analisis Selesai!")
                    st.markdown("---")
                    
                    # Kontainer untuk hasil
                    with st.container(border=True):
                        # DIMODIFIKASI: Menghilangkan "Langkah" dan membuat tulisan ke tengah
                        st.markdown("<h3 style='text-align: center; color: #1E88E5;'>Hasil Analisis Suara</h3>", unsafe_allow_html=True)
                        
                        col1, col2 = st.columns([0.7, 0.3])
                        with col1:
                            st.metric(label="Prediksi Utama", value=main_prediction_label, delta=f"Keyakinan: {main_prediction_prob:.2%}")
                        with col2:
                            if os.path.exists('temp_spec.png'):
                                st.image('temp_spec.png', caption='Spektogram')

                        st.markdown("##### Detail 5 Prediksi Teratas")
                        df_preds = pd.DataFrame(top_predictions, columns=["Kategori", "Probabilitas"])
                        st.dataframe(df_preds.style.format({"Probabilitas": "{:.2%}"}), use_container_width=True)
                        
                        st.bar_chart(df_preds.set_index("Kategori"))

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat analisis: {e}")
                finally:
                    if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)

    # --- FOOTER DIMODIFIKASI ---
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: grey; font-family: sans-serif;">
            <p>Project by:</p>
            <p style="font-weight: bold;">M.Nasyid Yunitian Rizal (22.11.5073)</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
