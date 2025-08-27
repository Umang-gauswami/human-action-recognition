
# ================================
# Step 1: Import OS and check classes
import os

# Path to dataset (adjust according to yours if you do locally)
data_dir = "/kaggle/input/human-action-recognition-har-dataset/Human Action Recognition"

# List all action classes
classes = os.listdir(data_dir)
# print("Number of classes:", len(classes))
# print("Classes (first 5 shown):", classes[:5])
# print(classes)

# Step 2: Load CSV files for train and test
import pandas as pd

train_csv = pd.read_csv(os.path.join(data_dir, "Training_set.csv"))
test_csv = pd.read_csv(os.path.join(data_dir, "Testing_set.csv"))

# print("Train shape:", train_csv.shape)
# print("Test shape:", test_csv.shape)
# print(train_csv.head())

# Step 3: Load and preprocess images by using opencv library
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Path of training images
train_img_dir = os.path.join(data_dir, "train")

# Resize dimension for CNN
IMG_SIZE = 128

# Load training images and labels
X = []
y = []
for i, row in train_csv.iterrows():
    img_path = os.path.join(train_img_dir, row['filename'])
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(row['label'])

X = np.array(X, dtype="float32") / 255.0  # normalize to 0-1
# print("X shape:", X.shape)

# Encode labels to categorical
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)
# print("y shape:", y_categorical.shape)
# print("Classes:", le.classes_)

# Step 4: Build base MobileNetV2 model with custom classifier
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

num_classes = len(le.classes_)

# Load pre-trained MobileNetV2
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base model initially
base_model.trainable = False

# Add custom classifier on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# model.summary()

# Step 5: Split data into training and validation

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)
# print("Train shape:", X_train.shape, "Validation shape:", X_val.shape)  # debug

# Step 6: Data augmentation, fine-tuning, and training
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# 1: Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
datagen.fit(X_train)

# 2: Fine-tune base model

base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False

# 3: Smaller classifier head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.6)(x)
x = Dense(
    64,
    activation='relu',
    kernel_regularizer=regularizers.l2(1e-4)
)(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
outputs = Dense(
    num_classes,
    activation='softmax',
    kernel_regularizer=regularizers.l2(1e-4)
)(x)

model = Model(inputs=base_model.input, outputs=outputs)

# 3: Compile with label smoothing
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss=loss_fn,
    metrics=['accuracy']
)
# 4: Callbacks
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss', patience=6, restore_best_weights=True, verbose=1
)

# 5: Train the model
history_finetune = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=50,
    callbacks=[reduce_lr, early_stop]
)

# Section 7: Save the trained model

model.save("trained-model.h5")
print("Model saved as trained-model.h5")
