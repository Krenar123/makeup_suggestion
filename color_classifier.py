import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

class ColorClassifier:
    def __init__(self, csv_path='datasets/colors.csv', test_size=0.2, random_state=42):
        self.csv_path = csv_path
        self.test_size = test_size
        self.random_state = random_state
        self.model = RandomForestClassifier()

        # Load data and perform initial setup
        self.setup()

    def setup(self):
        self.load_data()
        self.preprocess_data()
        self.train_model()
        self.evaluate_model()

    def load_data(self):
        # Read data from CSV file
        self.color_data = pd.read_csv(self.csv_path)

    def preprocess_data(self):
        # Preprocess the data
        X = self.color_data['color'].apply(self.hex_to_rgb).tolist()
        X = np.array(X) / 255.0  # Normalize RGB values
        y = LabelEncoder().fit_transform(self.color_data['type'])

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def train_model(self):
        # Train a RandomForestClassifier
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        # Make predictions on the test set
        y_pred = self.model.predict(self.X_test)

        # Evaluate the model
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f'Model Accuracy: {accuracy}')

    def predict_color_type(self, new_color_hex):
        # Example prediction
        new_color_rgb = np.array(self.hex_to_rgb(new_color_hex)) / 255.0

        # Use the fitted LabelEncoder for inverse_transform
        predicted_label = self.model.predict([new_color_rgb])
        label_encoder = LabelEncoder().fit(self.color_data['type'])
        predicted_type = label_encoder.inverse_transform(predicted_label)
        return [new_color_hex, predicted_type[0]]

    def hex_to_rgb(self, hex_color):
        # Function to convert hex to RGB
        hex_color = hex_color.lstrip('#')
        return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]

# Example Usage
#color_classifier = ColorClassifier()
#color_classifier.predict_color_type('#EFCCBE')  # Replace with your hex color
