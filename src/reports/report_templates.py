from typing import List, Dict

def generate_report(pred_probs: Dict[str, float], threshold: float = 0.5):
    # Simple template for binary classification (NORMAL vs PNEUMONIA)
    # Extend later for multi-label datasets.
    normal = pred_probs.get('NORMAL', 0.0)
    pneumonia = pred_probs.get('PNEUMONIA', 0.0)
    if pneumonia >= threshold:
        doctor = f"Findings indicate features consistent with pneumonia (confidence {pneumonia:.2f}). Clinical correlation recommended."
        patient = f"X-ray mein infection (pneumonia) ke asar dikh rahe hain. Mehrbani karke doctor se mashwara zaroor karein."
    else:
        doctor = f"No acute cardiopulmonary abnormality detected (confidence NORMAL {normal:.2f})."
        patient = f"Aapki X-ray theek lag rahi hai. Agar symptoms rahen to doctor se rabta karein."
    return {'doctor': doctor, 'patient': patient}
