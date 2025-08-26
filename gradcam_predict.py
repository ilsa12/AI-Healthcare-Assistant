import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

#  Paths
model_path = r"outputs\best_model.pth"   # trained model path
test_dir = r"data\test"                  # test folder
output_dir = r"gradcam_results"          # results save folder
os.makedirs(output_dir, exist_ok=True)

class_names = ["NORMAL", "PNEUMONIA"]

# Device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model 
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

# Transform 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Grad-CAM functions
def generate_gradcam(img_path, save_path):
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    gradients = []
    activations = []

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activation(module, input, output):
        activations.append(output)

    final_conv_layer = model.layer4[-1].conv2
    final_conv_layer.register_forward_hook(save_activation)
    final_conv_layer.register_full_backward_hook(save_gradient)

    # Prediction
    output = model(input_tensor)
    pred_class = torch.argmax(output, 1).item()

    # Grad-CAM
    model.zero_grad()
    score = output[0, pred_class]
    score.backward()

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

    img_np = np.array(img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    # Save output
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original X-ray")
    plt.imshow(img_np)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM Heatmap")
    plt.imshow(cam, cmap='jet')
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"Overlay ({class_names[pred_class]})")
    plt.imshow(superimposed_img)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Loop over test images 
for label in class_names:
    folder = os.path.join(test_dir, label)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        save_path = os.path.join(output_dir, f"{label}_{img_name}.png")
        generate_gradcam(img_path, save_path)
        print(f"✅ Saved: {save_path}")
