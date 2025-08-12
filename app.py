from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


app = Flask(__name__)
app.secret_key = 'your_secret_key'


UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


MODEL_PATH = 'glaucoma_model_mobilenetv2.h5'
model = load_model(MODEL_PATH)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_glaucoma(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    if prediction >= 0.5:
        return "🟢 Normal Eye", round(1 - prediction, 2)
    else:
        return "🔴 Glaucoma Detected", round(prediction, 2)
    


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result, confidence = predict_glaucoma(filepath)


        image_path = url_for('static', filename='uploads/' + filename)

        return render_template('result.html', image_path=image_path, result=result, confidence=confidence)

    else:
        flash('File type not allowed')
        return redirect(url_for('index'))


@app.route('/login.html')
def show_login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    if username == 'admin' and password == 'admin123':
        flash('Login successful!')
        return redirect(url_for('index'))
    else:
        flash('Invalid credentials. Try again.')
        return redirect(url_for('show_login'))


if __name__ == '__main__':
    app.run(debug=True)
