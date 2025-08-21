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

---

## Technical Implementation

### 1. Data Preparation
```python
# Data Augmentation

normal_files = [f for f, lbl in zip(train_generator.filenames, train_generator.classes) if lbl == 0]
pneumonia_files = [f for f, lbl in zip(train_generator.filenames, train_generator.classes) if lbl == 1]

normal_df = pd.DataFrame({"filename": normal_files, "class": 0})
pneumonia_df = pd.DataFrame({"filename": pneumonia_files, "class": 1})

normal_gen = normal_datagen.flow_from_dataframe(
    dataframe=normal_df,
    directory=train_generator.directory,
    target_size=(224, 224),
    x_col = 'filename',
    y_col = 'class',
    class_mode="raw",
    batch_size=16,  
    shuffle = True
)

pneumonia_gen = pneumonia_datagen.flow_from_dataframe(
    dataframe=normal_df,
    directory=train_generator.directory,
    target_size=(224, 224),
    x_col = 'filename',
    y_col = 'class',
    class_mode="raw",
    batch_size=16,
    shuffle = True,
)

# Generators

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),    # EfficientNet input size
    batch_size=32,
    class_mode='binary'        # 0/1 labels
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Transfer Learning with EfficientNetV2B0
base_model = EfficientNetV2B0(
    weights = "imagenet",
    include_top = False,
    input_shape = (224, 224, 3),
    pooling = None)

base_model.trainable = False

# Custom Classification Head
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Reduces spatial dimensions to 1x1
x = Dense(128, activation='swish')(x)
predictions = Dense(1, activation='sigmoid')(x)  # Binary output

model = Model(inputs=base_model.input, outputs=predictions)

# Class Weight Calculation
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes = np.unique(train_generator.classes),
    y = train_generator.classes
)
class_weights = {0: 2.0, 1: 1.0} #increased recall due to many false negatives 

#Used in training:
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    class_weight = class_weights
)

#Finding optimal decision threshold on validation set 
val_probs = model.predict(val_generator)
precision, recall, thresholds = precision_recall_curve(val_true, val_probs)
optimal_idx = np.argmin(np.abs(precision - recall))
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.3f}")

# C. Final evaluation on test data
test_probs = model.predict(test_generator)
y_pred = test_probs > optimal_threshold
print(classification_report(test_generator.classes, y_pred))
