# Pneumonia-Detection from Chest x-rays
I was bored, the project is for Pneumonia detection from chest x-rays using transfer learning with EfficientNetV2B0.
This is also my first time doing documentation, so it might be a bit messy sorry about that. 

## Project Overview
A deep learning system that classifies chest X-rays as **Normal (0)** or **Pneumonia (1)** using transfer learning with **EfficientNetV2B0**.  
The project addresses **class imbalance** and hopefully optimizes for clinical utility where detecting pneumonia (**recall**) is 
prioritized over reducing false alarms (**precision**).

## Dataset Structure (from kaggle)
chest_xray/
├── train/
│ ├── NORMAL/ # 1,341 images
│ └── PNEUMONIA/ # 3,875 images (bacterial/viral)
├── val/ # Validation set
└── test/ # Test set (624 images)

- **Class Imbalance:** ~1:3 ratio (Normal:Pneumonia) requiring special handling.
