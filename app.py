import os
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- تهيئة التطبيق ---
app = Flask(__name__)

# --- تحميل النموذج ---
MODEL_PATH = 'fire_detection_model_v2.keras'
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- أسماء الفئات ---
CLASS_NAMES = ['حريق (Fire)', 'لا يوجد حريق (No Fire)']

# --- دالة التنبؤ ---
def model_predict(img_path, model):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_batch = np.expand_dims(preprocessed_img, axis=0)
    predictions = model.predict(img_batch)
    score = np.max(predictions[0])
    predicted_class_index = np.argmax(predictions[0])
    return CLASS_NAMES[predicted_class_index], score


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'files' not in request.files:
            return render_template('index.html', error='لم يتم اختيار أي ملفات')

        files = request.files.getlist('files')

        if not files or all(file.filename == '' for file in files):
            return render_template('index.html', error='الرجاء اختيار صورة واحدة على الأقل')

        results = []
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'static', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)

        for file in files:
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                file_path = os.path.join(upload_folder, filename)
                file.save(file_path)

                prediction, score = model_predict(file_path, model)

                results.append({
                    'image': f'uploads/{filename}',
                    'prediction': prediction,
                    'score': f"{score * 100:.2f}%"
                })

        return render_template('index.html', results=results)

    return render_template('index.html')
