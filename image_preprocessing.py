import cv2
import glob
import numpy as np
import pickle

def preprocess_images(static_path, dynamic_path, target_size, output_dir):
    # Lists to store images and labels
    images = []
    labels = []

    # Load static images
    for img_path in glob.glob(static_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, target_size)
        images.append(img)
        labels.append(0)

    # Load dynamic images
    for img_path in glob.glob(dynamic_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, target_size)
        images.append(img)
        labels.append(1)

    # Convert lists to NumPy arrays
    X_train = np.array(images)
    y_train = np.array(labels)

    # Save X_train and y_train using pickle
    with open(output_dir + "/X_train.pkl", "wb") as f:
        pickle.dump(X_train, f)

    with open(output_dir + "/y_train.pkl", "wb") as f:
        pickle.dump(y_train, f)
