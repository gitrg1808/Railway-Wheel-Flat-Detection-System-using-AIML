import os
import shutil
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, send_from_directory

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define class labels
class_names = ["flat", "non_flat"]

# load Resnet model
def load_model(model_path):
    resnet = models.resnet18(pretrained=False)  
    num_features = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(num_features, 2), 
        nn.LogSoftmax(dim=1)
    )
    resnet.load_state_dict(torch.load(model_path, map_location=device))
    resnet = resnet.to(device)
    resnet.eval()  
    return resnet

model = load_model("resnet_classification5.0.h5")

# image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])

# predict class of image
def predict_image(image_path, model):
    try:
        
        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0)  
        input_tensor = input_tensor.to(device)

       
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)

        
        predicted_class = class_names[predicted.item()]
        return predicted_class

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# processing multiple images in a folder and copy flat images to a new folder
def predict_and_copy(folder_path, model, output_folder, save_results=False, results_file="results.txt"):
    results = []

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        if os.path.isfile(image_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            predicted_class = predict_image(image_path, model)
            if predicted_class:
                results.append((filename, predicted_class))
                print(f"{filename}: {predicted_class}")

                # Copy flat images to the new folder
                if predicted_class == "flat":
                    shutil.copy(image_path, os.path.join(output_folder, filename))

    # save results to a file if requested
    if save_results:
        with open(results_file, "w") as f:
            for filename, predicted_class in results:
                f.write(f"{filename}: {predicted_class}\n")
        print(f"Results saved to {results_file}")

    return results

# example usage
folder_path = "D:/KP/Wabtech/ExtractedImg"  #input folder path
output_folder = "D:/KP/Wabtech/FinalResult"  #desired output folder
results = predict_and_copy(folder_path, model, output_folder, save_results=True)


app = Flask(__name__)
UPLOAD_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# function to detect flat area and analyze severity
def detect_flat_area_and_severity(img, image_name, reference_mm=100.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=100, threshold2=200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_min, y_min, x_max, y_max = img.shape[1], img.shape[0], 0, 0

    for contour in contours:
        if cv2.contourArea(contour) >= 20:
            x, y, w, h = cv2.boundingRect(contour)
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x + w), max(y_max, y + h)

    if x_max > x_min and y_max > y_min:
        flat_area_mm2 = ((x_max - x_min) * (y_max - y_min)) * (reference_mm ** 2) / (img.shape[1] * img.shape[0])
        severity = calculate_severity(flat_area_mm2)
        impact_analysis = perform_impact_analysis(severity)

        # save original processed image
        processed_img_path = os.path.join(UPLOAD_FOLDER, f"processed_{image_name}")
        cv2.imwrite(processed_img_path, img)

        # heatmap
        heatmap_path = os.path.join(UPLOAD_FOLDER, f"heatmap_{image_name}")
        plt.figure(figsize=(6, 5))
        sns.heatmap(gray, cmap='jet', cbar=True)
        plt.axis('off')
        plt.savefig(heatmap_path)
        plt.close()

        return {
            "image_name": image_name,
            "image": f"processed_{image_name}",
            "heatmap": f"heatmap_{image_name}",
            "flat_area_mm2": round(flat_area_mm2, 2),
            "severity": severity,
            "impact_analysis": impact_analysis,
        }
    return None

# severity analysis
def calculate_severity(flat_area_mm2):
    if 1 <= flat_area_mm2 < 50:
        return "Low Severity"
    elif 50 <= flat_area_mm2 < 100:
        return "Medium Severity"
    elif flat_area_mm2 >= 100:
        return "High Severity"
    return "No Flat Area Detected"

# impact analysis
def perform_impact_analysis(severity):
    return {
        "Low Severity": "Minimal impact, normal operation.",
        "Medium Severity": "Potential impact, recommend further inspection.",
        "High Severity": "High risk, urgent replacement needed.",
        "No Flat Area Detected": "No impact."
    }.get(severity, "Unknown severity level.")

# process images in a folder
def process_folder(folder_path, reference_mm=100.0):
    results = []
    for image_file in os.listdir(folder_path):
        if image_file.lower().endswith(('png', 'jpg', 'jpeg')):
            img_path = os.path.join(folder_path, image_file)
            img = cv2.imread(img_path)
            if img is not None:
                result = detect_flat_area_and_severity(img, image_file, reference_mm)
                if result:
                    results.append(result)
    return results

@app.route('/')
def index():
    folder_path = 'D:/KP/Wabtech/FinalResult'  # Update path
    results = process_folder(folder_path)
    return render_template('index.html', results=results)

@app.route('/static/results/<filename>')
def get_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
