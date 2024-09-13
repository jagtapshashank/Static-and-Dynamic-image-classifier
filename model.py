import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Function to split the data into training and validation sets
def split_data(X_train, y_train, test_size=0.2, random_state=42):
    X_train1, X_val1, y_train1, y_val1 = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)
    return X_train1, X_val1, y_train1, y_val1

# Function to train the model
def train_model(X_train, y_train, X_val, y_val, num_epochs=30, batch_size=32):
    # Define a custom dataset
    class CustomDataset(tf.keras.utils.Sequence):
        def __init__(self, images, labels, batch_size, shuffle=True, augment=False):
            self.images = images
            self.labels = labels
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.augment = augment
            self.indexes = np.arange(len(self.images))
            if self.shuffle:
                np.random.shuffle(self.indexes)

            self.datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )

        def __len__(self):
            return int(np.ceil(len(self.images) / self.batch_size))

        def __getitem__(self, index):
            start = index * self.batch_size
            end = (index + 1) * self.batch_size
            batch_images = self.images[start:end]
            batch_labels = self.labels[start:end]

            if self.augment:
                # Apply data augmentation
                batch_images = self.apply_data_augmentation(batch_images)

            return np.array(batch_images), np.array(batch_labels)

        def apply_data_augmentation(self, batch_images):
            augmented_images = []
            for img in batch_images:
                augmented_img = self.datagen.random_transform(img)
                augmented_images.append(augmented_img)
            return augmented_images

    # Function to build VGG16-based DNN model
    def build_vgg16_dnn_augmented(num_classes):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(2048, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])

        return model

    # Instantiate the model
    num_classes = len(np.unique(y_train))
    model = build_vgg16_dnn_augmented(num_classes)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Create custom datasets for training and validation with data augmentation
    train_dataset = CustomDataset(X_train, y_train, batch_size, augment=True)
    val_dataset = CustomDataset(X_val, y_val, batch_size)

    # Training loop

    for epoch in range(num_epochs):
        # Training
        train_losses = []
        train_accuracies = []

        for inputs, labels in train_dataset:
            loss, accuracy = model.train_on_batch(inputs, labels)
            train_losses.append(loss)
            train_accuracies.append(accuracy)

        avg_train_loss = np.mean(train_losses)
        avg_train_accuracy = np.mean(train_accuracies)

        # Validation loop with accuracy and F1 score calculation
        val_losses = []
        val_accuracies = []
        val_f1_scores = []

        for val_inputs, val_labels in val_dataset:
            val_loss, val_accuracy = model.evaluate(val_inputs, val_labels, verbose=0)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # Predictions
            val_predictions = np.argmax(model.predict(val_inputs, verbose=0), axis=-1)

            # Convert true labels to one-hot encoding if needed
            if len(val_labels.shape) > 1:
                val_labels = np.argmax(val_labels, axis=-1)

            # Calculate F1 score
            f1 = f1_score(val_labels, val_predictions, average='weighted')
            val_f1_scores.append(f1)

        # Calculate average validation loss, accuracy, and F1 score
        avg_val_loss = np.mean(val_losses)
        avg_val_accuracy = np.mean(val_accuracies)
        avg_val_f1_score = np.mean(val_f1_scores)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}, Val F1 Score: {avg_val_f1_score:.4f}')

    # Print the validation accuracy and corresponding training accuracy after the last epoch
    print("Validation Accuracy after last epoch: ",  avg_val_accuracy)
    print("Corresponding Training Accuracy after last epoch: ",  avg_train_accuracy)

    return model

def save_model(model, model_file_path):
    # Save the model to a local pickle file
    with open(model_file_path, 'wb') as file:
        pickle.dump(model, file)
