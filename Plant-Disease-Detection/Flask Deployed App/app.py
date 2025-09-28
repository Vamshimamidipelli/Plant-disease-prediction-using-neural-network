import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd

# Read CSV files for disease and supplement info
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load the pre-trained model
model = CNN.CNN(39)
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    try:
        image = Image.open(image_path)
        image = image.resize((224, 224))
        input_data = TF.to_tensor(image)
        input_data = input_data.view((-1, 3, 224, 224))
        output = model(input_data)
        output = output.detach().numpy()
        index = np.argmax(output)
        return index
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/team')
def team_page():
    return render_template('team.html')

@app.route('/contact')
def contact_page():
    return render_template('contact.html')

@app.route('/ai-tools')
def ai_tools_page():
    return render_template('ai_tools.html')

@app.route('/nutrition')
def nutrition_page():
    return render_template('nutrition.html')

@app.route('/guides')
def guides_page():
    return render_template('guides.html')

@app.route('/ai_projects')
def ai_projects_page():
    return render_template('ai_projects.html')

@app.route('/overview')
def overview_page():
    return render_template('overview.html')

@app.route('/features')
def features_page():
    return render_template('features.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        if 'image' not in request.files or request.files['image'].filename == '':
            # No image provided
            return render_default_page()
        else:
            # Image provided
            image = request.files['image']
            filename = image.filename
            file_path = os.path.join('static/uploads', filename)
            image.save(file_path)
            print(file_path)
            pred = prediction(file_path)
            if pred is None or pred >= len(disease_info):
                # Prediction failed or out of bounds
                return render_default_page()
            title = disease_info['disease_name'][pred]
            description = disease_info['description'][pred]
            prevent = disease_info['Possible Steps'][pred]
            image_url = disease_info['image_url'][pred]
            supplement_name = supplement_info['supplement name'][pred]
            supplement_image_url = supplement_info['supplement image'][pred]
            supplement_buy_link = supplement_info['buy link'][pred]
            return render_template('submit.html', title=title, desc=description, prevent=prevent,
                                   image_url=image_url, pred=pred, sname=supplement_name, simage=supplement_image_url, buy_link=supplement_buy_link)

def render_default_page():
    default_image_path = 'static/default.jpg'  # Path to the default image
    default_text = "This is a default page. Please upload a valid plant image for analysis."
    return render_template('submit.html', title="Default Title", desc=default_text, prevent="",
                           image_url=default_image_path, pred="", sname="", simage="", buy_link="")

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']), disease=list(disease_info['disease_name']), buy=list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
