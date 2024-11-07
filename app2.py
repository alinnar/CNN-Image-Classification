from PIL import Image
import io
import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_option_menu import option_menu 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Membuat objek ImageDataGenerator untuk augmentasi data
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

st.set_page_config(page_title="Face Detection Classification")

@st.cache_resource
def load_model_data():
    model = tf.keras.models.load_model('model2.h5')
    class_names = ['angry', 'happy', 'nothing', 'sad']
    return model, class_names

def beranda():
    st.title('Face Detection Classification🎭')
    st.markdown('''Selamat datang di **Face Detection Classification!** Aplikasi ini mendeteksi dan mengklasifikasikan wajah 
                berdasarkan ekspresi emosi atau kategori lainnya.''')
    
    st.markdown('''
                Cara Kerja Aplikasi 
                1. **Deteksi Wajah**: Mengidentifikasi lokasi wajah dalam gambar.
                2. **Preprocessing Wajah**: Memotong dan mengubah ukuran wajah agar sesuai dengan input model.
                3. **Klasifikasi Wajah**: CNN menganalisis dan mengklasifikasikan wajah.
                4. **Tampilan Hasil**: Hasil klasifikasi ditampilkan langsung di halaman ini.
                ''')

def klasifikasi(model, class_names):
    st.markdown("## Mulai Deteksi Wajah Anda!")
    st.text("Unggah gambar atau mulai streaming video di bawah ini.")

    st.session_state['my_img'] = None
    treshold = 0.6
    option = st.selectbox("Pilih pengambilan gambar", ["Upload foto", "Ambil foto"])

    if option == "Upload foto":
        uploaded_file = st.file_uploader('Upload gambar wajahmu', type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            pil_image = Image.open(io.BytesIO(uploaded_file.read())).convert('RGB')
            st.session_state['my_img'] = pil_image
            st.image(pil_image)
            pil_image = pil_image.resize((150, 150))
            pil_image = np.array(pil_image) / 255.0
            prediction = model.predict(np.expand_dims(pil_image, axis=0))

            index = np.argmax(prediction)
            confidence = prediction[0][index]

            if confidence >= treshold:
                st.success(f"Prediksi: {class_names[index]} {confidence * 100:.2f}%")
                solution(class_names[index], class_names, confidence)
            else:
                st.warning("Emosi tidak ditemukan.")

    elif option == "Ambil foto":
        image_captured = st.camera_input('')
        if image_captured:
            st.session_state['my_img'] = image_captured

        if st.session_state['my_img']:
            img_byte_array = st.session_state['my_img'].getvalue()
            img = Image.open(io.BytesIO(img_byte_array)).resize((150, 150))
            img = np.array(img) / 255.0
            prediction = model.predict(np.expand_dims(img, axis=0))
            index = np.argmax(prediction)
            confidence = prediction[0][index]

            if confidence >= treshold:
                st.success(f"Prediksi: {class_names[index]} {confidence * 100:.2f}%")
                solution(class_names[index], class_names, confidence)
            else:
                st.warning("Emosi tidak ditemukan.")

def solution(nama_kelas_index, nama_kelas, confidence):
    emosi_dict = {
        nama_kelas[0]: "angry",
        nama_kelas[1]: "happy",
        nama_kelas[2]: "nothing",
        nama_kelas[3]: "sad"
    }
    st.write(f"Prediksi: {emosi_dict.get(nama_kelas_index, 'Unknown')} {confidence * 100:.2f}%")

def Tentang():
    st.title('Aplikasi Face Detection and Classification')
    st.markdown('''Aplikasi ini menggunakan deep learning, khususnya CNN, untuk mendeteksi dan mengklasifikasikan wajah 
                sesuai dengan kategori yang diinginkan.''')

def main():
    st.markdown("""<style>/* Custom styling */</style>""", unsafe_allow_html=True)

    model, class_names = load_model_data()

    selected = option_menu(None, ['Beranda', 'Deteksi Emosi dari Wajah', 'Tentang'],
        icons=['house', 'patch-question-fill', 'file-earmark-person'],
        menu_icon='cast', default_index=0, orientation='horizontal')

    if selected == 'Beranda':
        beranda()
    elif selected == 'Deteksi Emosi dari Wajah':
        klasifikasi(model, class_names)
    elif selected == 'Tentang':
        Tentang()

if __name__ == "__main__":
    main()