# AI4ALL Fellowship Fall 2024 - Pneumonia Diagnosis

## Project Proposal

### Research Question  
To what extent can pediatric pneumonia diagnosis be accelerated using machine learning?

---

### Summary  
Pneumonia is an infection that “inflames the air sacs in one or both lungs, typically with fluid or pus, causing coughing, fever, chills, and difficulty breathing.”  
- It is life-threatening to infants, children, and people aged 65+.  
- Different types of pneumonia (bacterial, viral, etc.) require different forms of treatment.  
- Pediatric pneumonia is a leading cause of death in children under the age of 5 (Ebdeldike), with 1.3 million deaths per year.

The two main causes of pneumonia are bacteria and viruses, but they are treated very differently ([Source](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0092867418301545%3Fshowall%3Dtrue#)).
- **Bacterial pneumonia** requires urgent antibiotics, while **viral pneumonia** requires supportive care. 
- Therefore, speedy diagnoses are important. 

One way to diagnose patients is with X-rays, which require a doctor to interpret the scan. We hope our project will accelerate diagnosis by reducing the need for someone to interpret the scans.

We plan to create a computer vision model that can predict whether someone has the disease based on a chest X-ray. We are using Daniel Kermany, Kang Zhang, and Michael Goldbaum’s dataset “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification,” which contains thousands of pediatric X-ray scans classified as viral pneumonia, bacterial pneumonia, or normal.

This model is an example of using machine learning to help improve healthcare.

---

## Model & Results

### Data Preparation
- Resized images to 224x224  
- Normalized to ImageNet mean and standard deviation  
- Applied random flips and rotations

### Model: EfficientNetB0
- **Background**: Trained on the ImageNet database, capable of classifying into 1000 object categories.  
- **Fine-tuned** for binary classification (healthy or pneumonia diagnosis).

### Results
- **88.8% Accuracy**  
- **28% False Positive Rate**  
- **1% False Negative Rate**

[Final Presentation Slides](https://docs.google.com/presentation/d/149CnEzX5nG3UWlited4vjbrFapRgfA0XBAHYKpOO9cQ/edit?usp=sharing)  
[Kaggle Dataset Link](https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray)