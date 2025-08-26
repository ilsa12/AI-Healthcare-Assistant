import torch
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.reshape_transforms import torchvision_resnet50_reshape_transform
from pytorch_grad_cam.utils.image import preprocess_image
import numpy as np
import os

def generate_gradcam(model, image_path: str, class_idx: int = None,
                     layer_name: str = 'layer4', save_path: str = 'outputs/heatmaps/gradcam.png'):
    model.eval()
    # pick target layer
    target_layer = getattr(model, layer_name)
    # load image (assumes RGB or grayscale convertible to RGB)
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    rgb_img = np.float32(img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    targets = None
    if class_idx is not None:
        targets = [ClassifierOutputTarget(class_idx)]
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(visualization).save(save_path)
    return save_path
