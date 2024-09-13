import argparse
import pickle
import image_preprocessing
import model
import os
import cv2
import glob
import numpy as np
import pandas as pd

def preprocess_images(static_path, dynamic_path, target_size, output_dir):
    image_preprocessing.preprocess_images(static_path, dynamic_path, target_size, output_dir)
    print("Image preprocessing completed and pickle files saved.")

def train_model(X_train_file, y_train_file, output_dir):
    # Load X_train and y_train from pickle files
    with open(X_train_file, 'rb') as f:
        X_train = pickle.load(f)

    with open(y_train_file, 'rb') as f:
        y_train = pickle.load(f)

    # Split the data
    X_train1, X_val1, y_train1, y_val1 = model.split_data(X_train, y_train)
    # Train the model
    trained_model = model.train_model(X_train1, y_train1, X_val1, y_val1)
    # Save the trained model
    model.save_model(trained_model, output_dir + "/trained_model.pkl")
    print("Model training completed and pickle file for the model saved.")

def test_model(test_images_folder, model_file):
    # Change directory to the folder containing test images
    os.chdir(test_images_folder)

    # Load images and assign labels
    pic = glob.glob('*.jpg')

    images = []

    # Resize images to a common size
    target_size = (224, 224)

    for img_path in pic:
        img = cv2.imread(img_path)
        img = cv2.resize(img, target_size)
        images.append(img)

    # Convert lists to NumPy arrays
    X_test = np.array(images)

    # Load the trained model from the pickle file
    with open(model_file, 'rb') as file:
        model = pickle.load(file)

    # Make predictions
    predictions = model.predict(X_test)

    # Get the predicted class for each image by taking the index with the maximum probability
    predicted_classes = np.argmax(predictions, axis=1)

    # Assuming predictions has 2 values for each image
    static_predictions = predictions[:, 0]
    dynamic_predictions = predictions[:, 1]

    # Create a DataFrame with Image name, predicted_classes, and split predictions
    data = {'Image name': pic, 'predicted_classes': predicted_classes, 'static_prediction_prob': static_predictions, 'dynamic_prediction_prob': dynamic_predictions}
    df = pd.DataFrame(data)

    # Save the DataFrame to an Excel file
    excel_file_path = os.path.join(os.path.dirname(test_images_folder), 'predictions.xlsx')
    df.to_excel(excel_file_path, index=False)

    print("Excel file saved at:", excel_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script for image preprocessing, model training, and testing.")
    parser.add_argument("task", choices=["preprocess", "train", "test"], help="Specify the task to perform.")
    parser.add_argument("--static_path", help="Path to the directory containing static images.")
    parser.add_argument("--dynamic_path", help="Path to the directory containing dynamic images.")
    parser.add_argument("--target_size", nargs=2, type=int, help="Target size for image resizing.")
    parser.add_argument("--output_dir", help="Directory to save output files.")
    parser.add_argument("--X_train_file", help="Path to the X_train pickle file.")
    parser.add_argument("--y_train_file", help="Path to the y_train pickle file.")
    parser.add_argument("--test_images", help="Path to the directory containing test images.")
    parser.add_argument("--model_file", help="Path to the trained model pickle file.")

    args = parser.parse_args()

    if args.task == "preprocess":
        preprocess_images(args.static_path, args.dynamic_path, tuple(args.target_size), args.output_dir)
    elif args.task == "train":
        train_model(args.X_train_file, args.y_train_file, args.output_dir)
    elif args.task == "test":
        test_model(args.test_images, args.model_file)
