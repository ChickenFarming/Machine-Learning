# -*- coding: utf-8 -*-
"""Capstone.Project1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LIZ0Dvcn1yhlrd6NNuiYHS9XMiP5Zvov
"""

# Load library
import os
import time
import shutil
from shutil import copyfile
import pathlib
import itertools
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras import layers, models

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

print ('modules loaded')

from google.colab import drive
drive.mount('/content/drive')

# Define the paths to your training and validation datasets
dataset_dir = '/content/drive/MyDrive/Dataset/Train'
csv_dir = '/content/drive/MyDrive/Dataset/train_data.csv'

"""**Membuat Fungsi Split Data, Pembangkit Gambar, Penampil Gambar, Callback, Plot, dan Matrik Konfusion**"""

# Split dataframe into train, valid, and test sets
def split_data(dataset_dir, csv_dir):
    # Read CSV file into a DataFrame
    df = pd.read_csv(csv_dir)

    # Rename columns for better clarity
    df.columns = ['filepaths', 'labels']

    # Update filepaths by joining dataset_dir with the existing filepaths
    df['filepaths'] = df['filepaths'].apply(lambda x: os.path.join(dataset_dir, x))

    penyakit = df['labels']
    # Create the training dataframe
    # Use stratify based on 'labels' for balanced class distribution
    train_df, dummy_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=123, stratify=penyakit)

    penyakit = dummy_df['labels']
    # Further split the remaining dataframe into validation and test sets
    valid_df, test_df = train_test_split(dummy_df, train_size=0.5, shuffle=True, random_state=123, stratify=penyakit)

    return train_df, valid_df, test_df

train_df, valid_df, test_df = split_data(dataset_dir, csv_dir)

"""**Data Generator**"""

def create_generators(train_df, valid_df, test_df, batch_size, class_mode, target_size):
    """
    Create image data generators for training, validation, and testing.

    Parameters:
    - train_df: DataFrame for training data
    - valid_df: DataFrame for validation data
    - test_df: DataFrame for test data
    - batch_size: Batch size for training and validation

    Returns:
    - train_gen: Image data generator for training
    - valid_gen: Image data generator for validation
    - test_gen: Image data generator for testing
    """
    # Determine test batch size dynamically based on the length of the test set
    ts_length = len(test_df)
    test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length % n == 0 and ts_length / n <= 80]))
    test_steps = ts_length // test_batch_size

        #augmentasi data
    def scalar(img):
        return img

    # Image data generators
    tr_gen = ImageDataGenerator(preprocessing_function=scalar, horizontal_flip=True)
    ts_gen = ImageDataGenerator(preprocessing_function=scalar)

    # Create generators from dataframes
    train_gen = tr_gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=target_size,
                                          class_mode=class_mode, shuffle=True, batch_size=batch_size)

    valid_gen = ts_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=target_size,
                                          class_mode=class_mode, shuffle=True, batch_size=batch_size)

    # Use custom test_batch_size and disable shuffling for testing
    test_gen = ts_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=target_size,
                                          class_mode=class_mode, shuffle=False, batch_size=test_batch_size)

    return train_gen, valid_gen, test_gen

# Get generator
train_gen, valid_gen, test_gen = create_generators(train_df, valid_df, test_df, batch_size=40, class_mode='categorical', target_size=(224,224))

def create_model(input_shape=(224, 224, 3), num_classes=4):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(axis=-1),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(axis=-1),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(axis=-1),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
        )

    return model

batch_data, batch_labels = next(train_gen)
print("Shape of batch data:", batch_data.shape)
print("Shape of batch labels:", batch_labels.shape)

model = create_model()
model.summary()

history = model.fit(x= train_gen, epochs= 15, verbose= 1,
                    validation_data= valid_gen, validation_steps= None, shuffle= False)

plot_training_history(history)