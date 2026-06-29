from flask import Flask, render_template, request, send_from_directory
import os
from datetime import datetime
import hashlib
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "resnet50.h5"
CLASS_NAME_PATH = BASE_DIR / "model" / "class_names.pkl"
CLASS_DETAIL = BASE_DIR / "static" / "class_names.xlsx"

# Menentukan folder untuk menyimpan file yang diupload
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = tf.keras.models.load_model(MODEL_PATH)

class_detail = pd.read_excel(CLASS_DETAIL)

with open(CLASS_NAME_PATH, 'rb') as f:
    class_names = pickle.load(f)

# Pastikan folder 'uploads' ada, jika belum, maka buat
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html', show_result=False)

def encrypt_filename(filename):
    # Menggunakan waktu saat ini dan nama asli file untuk memastikan keunikan
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    hash_object = hashlib.md5((filename + timestamp).encode())
    # Mendapatkan nama file terenkripsi dengan ekstensi asli
    encrypted_name = hash_object.hexdigest() + os.path.splitext(filename)[1]
    return encrypted_name

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST': 
        f = request.files['image']

        # Meng-enkripsi nama file
        encrypted_filename = encrypt_filename(f.filename)
        
        # Mendapatkan path lengkap untuk menyimpan file terenkripsi
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], encrypted_filename)
        
        # Menyimpan file di dalam folder yang ditentukan
        f.save(filepath)

        img = tf.keras.utils.load_img(f"uploads/{encrypted_filename}", target_size=(150,150))
        img_array = tf.keras.utils.img_to_array(img)
        img_scale = img_array/255
        batch_image_array = np.array([img_scale])
        result = model.predict(batch_image_array)
        test_pred_classes = np.argmax(result, axis=1)
        class_name = class_names[test_pred_classes[0]]
        probability = np.round(100 * np.max(result[0]), 2)

        detail_df = class_detail[class_detail['id'] == test_pred_classes[0]]
        detail = ""
        for i in detail_df['detail']:
            detail = i
              
    return render_template('index.html', show_result=True, namafile=encrypted_filename, class_name=class_name, probability=probability, detail=detail)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)