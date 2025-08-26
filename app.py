import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Model load
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["NORMAL", "PNEUMONIA"]

model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("outputs/best_model.pth", map_location=device))
model.eval()
model.to(device)

# Transform for input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Gradcam helpers
def generate_gradcam(model, image_path, save_prefix):
    """Generate heatmap overlay and bounding box image"""

    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Hook
    final_conv_layer = model.layer4[-1].conv2
    gradients = []
    activations = []

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activation(module, input, output):
        activations.append(output)

    final_conv_layer.register_forward_hook(save_activation)
    final_conv_layer.register_full_backward_hook(save_gradient)

    # Forward + backward
    output = model(input_tensor)
    pred_class = torch.argmax(output, 1).item()
    score = output[0, pred_class]
    model.zero_grad()
    score.backward()

    # Grad-CAM heatmap
    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i, :, :]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    # Convert images
    img_np = np.array(img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    # Bounding Box from heatmap
    thresh = np.uint8(cam > 0.5) * 255
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box_img = overlay.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(box_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Save results
    original_path = os.path.join(RESULT_FOLDER, f"{save_prefix}_original.jpg")
    heatmap_path = os.path.join(RESULT_FOLDER, f"{save_prefix}_heatmap.jpg")
    box_path = os.path.join(RESULT_FOLDER, f"{save_prefix}_box.jpg")

    cv2.imwrite(original_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(heatmap_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(box_path, cv2.cvtColor(box_img, cv2.COLOR_RGB2BGR))

    return class_names[pred_class], original_path, heatmap_path, box_path

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Run Grad-CAM
            pred, original_img, heatmap_img, box_img = generate_gradcam(
                model, filepath, os.path.splitext(file.filename)[0]
            )

            return render_template("index.html",
                                   prediction=pred,
                                   original_image="/" + original_img,
                                   overlay_heatmap="/" + heatmap_img,
                                   overlay_box="/" + box_img)

    return render_template("index.html")

# Run
if __name__ == "__main__":
    app.run(debug=True)
