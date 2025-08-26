# 🩺 AI Healthcare Assistant (Chest X-ray Classifier with Grad-CAM)

This project is a deep learning-based web application that helps radiologists detect Pneumonia from chest X-ray images.
It uses ResNet as the backbone model and integrates Grad-CAM for explainability (highlighting affected lung regions).

## 🚀 Features
- Upload chest X-rays through a web interface.
- Predict whether the scan is **Normal** or **Pneumonia**.
- Automatically generate **Grad-CAM heatmaps** to explain predictions.
- Bounding box overlay to highlight the most affected regions.
- Lightweight Flask-based web app for doctors.

---

## 📂 Project Structure
AI_Healthcare_Assistant/
│── app.py # Flask app
│── outputs/ # trained models (best_model.pth here)
│── static/uploads/ # uploaded scans
│── templates/index.html # frontend UI
│── src/ # model, utils, reports
│── requirements.txt # dependencies
│── README.md # documentation


---

## ⚡ Quickstart (Developers)

### 1. Create & activate a virtual environment
**Windows (PowerShell)**
```bash
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

2. Install dependencies
pip install -r requirements.txt

3. Download Trained Model

Since GitHub doesn’t allow large files, download best_model.pth from the link below and place it in the outputs/ folder:https://drive.google.com/uc?id=1Wgc9vFEDKNxEeDvnaTKxDXktkHZ6vULQ&export=download

4. Run the Web App
python app.py
Then open http://127.0.0.1:5000 in your browser.

To generate Grad-CAM for a test image:
python gradcam_predict.py --image-path "test_images/person1_virus_6.jpeg"



