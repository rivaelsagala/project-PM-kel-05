from flask import Flask, render_template, request, redirect, url_for, flash
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import threading

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'
app.secret_key = 'your_secret_key'

# Global variables
camera_thread = None
stop_camera = False

MODEL_CONFIGS = {
    'model_1': {
        'name': 'CNN ReLU',
        'path': '../../model/model_CNN_relu.keras',
        'labels': ['Mujahir', 'Red Devil', 'Sepat']
    },
    'model_2': {
        'name': 'CNN Tanh',
        'path': '../../model/model_CNN_tanh_object.keras',
        'labels': ['Mujahir', 'Red Devil', 'Sepat']
    },
    'model_3': {
        'name': 'ANN ReLU',
        'path': '../../model/model_ANN_relu.keras',
        'labels': ['Mujahir', 'Red Devil', 'Sepat']
    },
    'model_4': {
        'name': 'ANN Tanh',
        'path': '../../model/model_ANN_tanh_object.keras',
        'labels': ['Mujahir', 'Red Devil', 'Sepat']
    }
}

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_frame(frame, model, labels):
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)
    
    predictions = model.predict(input_frame)
    predicted_label = labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_label, confidence

def camera_function(model_id):
    global stop_camera
    
    # Load the selected model
    model = load_model(MODEL_CONFIGS[model_id]['path'])
    labels = MODEL_CONFIGS[model_id]['labels']
    
    # Start camera
    cap = cv2.VideoCapture(0)
    
    while not stop_camera:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Make prediction
        predicted_label, confidence = predict_frame(frame, model, labels)
        
        # Add text to frame
        text = f"{predicted_label}: {confidence:.2f}%"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Fish Classification', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    stop_camera = False

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/start_camera/<model_id>")
def start_camera(model_id):
    global camera_thread, stop_camera
    
    if model_id in MODEL_CONFIGS:
        if camera_thread is None or not camera_thread.is_alive():
            stop_camera = False
            camera_thread = threading.Thread(target=camera_function, args=(model_id,))
            camera_thread.start()
            return "Camera started"
    return "Invalid model"

@app.route("/stop_camera")
def stop_camera():
    global stop_camera
    stop_camera = True
    return "Camera stopped"

@app.route("/klasifikasi", methods=["GET", "POST"])
def klasifikasi():
    if request.method == "POST":
        selected_model = request.form.get('selectedModel')
        
        if not selected_model or selected_model not in MODEL_CONFIGS:
            flash("Silakan pilih model yang valid.")
            return redirect(request.url)

        file = request.files.get('image')
        if not file:
            flash("Silakan unggah gambar.")
            return redirect(request.url)

        filename = file.filename
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(img_path)
        
        try:
            model = load_model(MODEL_CONFIGS[selected_model]['path'])
            img = preprocess_image(img_path)
            predictions = model.predict(img)
            
            # Get all class probabilities
            class_probabilities = []
            for i, prob in enumerate(predictions[0]):
                class_probabilities.append({
                    "label": MODEL_CONFIGS[selected_model]['labels'][i],
                    "probability": round(float(prob * 100), 2)
                })
            
            # Sort probabilities to find the highest
            class_probabilities.sort(key=lambda x: x["probability"], reverse=True)
            
            result = {
                "jenis_ikan": class_probabilities[0]["label"],
                "akurasi": class_probabilities[0]["probability"],
                "image_filename": filename,
                "model_name": MODEL_CONFIGS[selected_model]['name'],
                "all_probabilities": class_probabilities
            }
            return render_template("klasifikasi.html", result=result, models=MODEL_CONFIGS)

        except Exception as e:
            flash(f"Error during classification: {str(e)}")
            return redirect(request.url)

    return render_template("klasifikasi.html", result=None, models=MODEL_CONFIGS)

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)