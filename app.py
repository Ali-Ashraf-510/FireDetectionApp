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

# --- تحميل النموذج المُدرب مسبقًا ---
MODEL_PATH = 'fire_detection_model.h5'
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- تعريف أسماء الفئات ---
# يجب أن تكون بنفس ترتيب الفئات التي تدرب عليها النموذج
# في حالتنا: fire_images, non_fire_images
# تأكد من الترتيب الصحيح من خلال train_images.class_indices في دفتر Colab
CLASS_NAMES = ['حريق (Fire)', 'لا يوجد حريق (No Fire)']


def model_predict(img_path, model):
    """
    دالة لمعالجة الصورة والتنبؤ بنوعها
    :param img_path: مسار الصورة
    :param model: النموذج المحمل
    :return: اسم الفئة المتوقعة ونسبة الثقة
    """
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    
    # استخدام دالة المعالجة الخاصة بنموذج MobileNetV2
    preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # إضافة بعد إضافي لتناسب إدخال النموذج
    img_batch = np.expand_dims(preprocessed_img, axis=0)

    # إجراء التنبؤ
    predictions = model.predict(img_batch)
    
    # الحصول على الفئة ذات الاحتمالية الأعلى ونسبة الثقة
    score = np.max(predictions[0])
    predicted_class_index = np.argmax(predictions[0])
    
    # إرجاع اسم الفئة ونسبة الثقة
    return CLASS_NAMES[predicted_class_index], score


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='لم يتم اختيار أي ملف')
        
        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='لم يتم اختيار أي ملف')

        if file:
            filename = secure_filename(file.filename)
            
            # إنشاء مسار لحفظ الملف المرفوع في مجلد 'static/uploads'
            basepath = os.path.dirname(__file__)
            upload_folder = os.path.join(basepath, 'static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, filename)
            
            file.save(file_path)

            # إجراء التنبؤ
            prediction, score = model_predict(file_path, model)

            # إرسال النتيجة ومسار الصورة إلى الواجهة الأمامية
            return render_template('index.html', 
                                   prediction=prediction, 
                                   score=f"{score*100:.2f}%",
                                   uploaded_image=f'uploads/{filename}')
    
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
