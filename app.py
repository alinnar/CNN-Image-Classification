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

# Mengatur jalur ke direktori dataset Anda
train_directory = 'path_to_your_training_data'

# Menghitung jumlah gambar untuk setiap kelas
class_counts = {
    'anger': 26,
    'happy': 19,
    'sad': 22
}

# Tentukan jumlah target untuk setiap kelas (kelas dengan jumlah terbanyak)
target_count = max(class_counts.values())

# Oversampling untuk kelas yang kurang
for class_label, count in class_counts.items():
    if count < target_count:
        # Mengambil direktori spesifik untuk kelas
        class_directory = f"{train_directory}/{class_label}"
        
        # Memuat gambar dari direktori kelas
        generator = datagen.flow_from_directory(
            train_directory,
            classes=[class_label],
            target_size=(224, 224),  # Ubah ukuran sesuai kebutuhan
            batch_size=32,
            class_mode='categorical',
            shuffle=True
        )
        
        # Menghitung jumlah tambahan yang diperlukan
        additional_samples_needed = target_count - count
        
        # Menghasilkan dan menyimpan gambar tambahan
        for i in range(additional_samples_needed // 32 + 1):
            x_batch, y_batch = next(generator)
            # Simpan x_batch dan y_batch ke dalam dataset Anda di sini
            # Misalnya: save_to_dataset(x_batch, y_batch)


st.set_page_config(page_title="Face Detection Classification")

@st.cache_resource
def load_model_data():
    model = tf.keras.models.load_model('model.h5')
    class_names = ['sad', 'happy', 'anger']
    return model, class_names

def beranda():

    st.title('Face Detection Classification🎭')

    st.markdown('''
    Selamat datang di **Face Detection Classification!** Aplikasi ini memungkinkan Anda mendeteksi wajah dan mengklasifikasikannya secara otomatis 
                berdasarkan kategori tertentu, seperti ekspresi emosi atau identitas.''')
    
    st.markdown('''
                Cara Kerja Aplikasi 
                1. Deteksi Wajah
                   Aplikasi ini menggunakan model pendeteksian wajah yang telah dilatih untuk mengidentifikasi lokasi wajah dalam gambar atau video.

                2. Preprocessing Wajah
                   Setelah wajah terdeteksi, gambar wajah akan dipotong dan diubah ukurannya agar sesuai dengan input yang diperlukan oleh model klasifikasi.

                3. Klasifikasi Wajah
                   Model Convolutional Neural Network (CNN) kemudian menganalisis wajah tersebut dan menentukan kategori yang paling sesuai, misalnya emosi (senang, sedih, marah) atau kategori lainnya.

                4. Tampilan Hasil
                   Hasil deteksi dan klasifikasi akan ditampilkan langsung di halaman ini, memberikan Anda gambaran langsung mengenai hasil analisis wajah.''')

def klasifikasi(model, class_names):
    st.markdown("## Mulai Deteksi Wajah Anda!")
    st.text('''Unggah gambar atau mulai streaming video di bawah ini untuk mencoba fitur deteksi 
dan klasifikasi wajah kami secara langsung.''')

    st.session_state['my_img'] = None
    treshold = 0.6

    option = st.selectbox("Pilih pengambilan gambar", ["Upload foto", "Ambil foto"])

    if option == "Upload foto":
        uploaded_file = st.file_uploader('Upload gambar wajahmu', type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            pil_image = Image.open(io.BytesIO(uploaded_file.read())).convert('RGB')
            st.session_state['my_img'] = pil_image
            st.image(pil_image)

            pil_image = pil_image.resize((224, 224))
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
            img = Image.open(io.BytesIO(img_byte_array)).resize((224, 224))
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
        nama_kelas[0]: "sad",
        nama_kelas[1]: "happy",
        nama_kelas[2]: "anger"
    }
    st.write(f"Prediksi: {emosi_dict.get(nama_kelas_index, 'Unknown')} {confidence * 100:.2f}%")

def Tentang():
    st.title('Aplikasi Face Detection and Classification')
    st.markdown('''Aplikasi Face Detection and Classification ini dikembangkan sebagai solusi cerdas untuk mendeteksi dan 
                mengklasifikasikan wajah dalam gambar atau video secara real-time. 
                Dengan memanfaatkan teknologi deep learning, khususnya model Convolutional Neural Network (CNN), 
                aplikasi ini mampu mengenali wajah, menganalisis ekspresi emosi, atau mengidentifikasi fitur wajah tertentu sesuai kategori yang diinginkan.
                ''')

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
