from __future__ import print_function
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

class DataGenerator:
    def __init__(self, datapath):
        """
        A class to generate and manage data for image processing tasks.

        Args:
            datapath (str): The directory where dataset files are stored.
        """
        self.data_path = Path(datapath)

    def split_validation_data(self, val_filename='val'):
        """
        Splits the original training dataset into train and validation sets.

        Args:
            val_filename (str): The name for the validation file to be created.
        """
        train_csv_path = self.data_path / 'train.csv'

        # Check if the training CSV file exists
        if not train_csv_path.exists():
            raise FileNotFoundError(f"Training CSV file not found at: {train_csv_path}")

        # Read the training data
        train_data = pd.read_csv(train_csv_path)

        # Split the data into validation and training sets
        validation_data = train_data.iloc[:3589]
        train_data = train_data.iloc[3589:]

        # Save the new datasets to CSV files
        validation_data.to_csv(self.data_path / f'{val_filename}.csv', index=False)
        train_data.to_csv(self.data_path / 'train.csv', index=False)

        print(f"Data split complete: {len(validation_data)} validation samples and {len(train_data)} train samples.")

    def str_to_image(self, pixel_str):
        """
        Converts a string of pixel values into a PIL image object.

        Args:
            pixel_str (str): A string of space-separated pixel values.

        Returns:
            PIL.Image: A PIL Image object.
        """
        pixel_values = np.array(pixel_str.split(), dtype=np.uint8).reshape(48, 48)
        return Image.fromarray(pixel_values)

    def save_images(self, datatype='train'):
        """
        Saves images from the given dataset ('train', 'val', or 'test') to a folder.

        Args:
            datatype (str): The dataset type ('train', 'val', or 'test').
        """
        folder_path = self.data_path / datatype
        csv_file_path = self.data_path / f'{datatype}.csv'

        # Create the folder if it doesn't exist
        folder_path.mkdir(parents=True, exist_ok=True)

        # Check if the CSV file exists
        if not csv_file_path.exists():
            raise FileNotFoundError(f"{datatype.capitalize()} CSV file not found at: {csv_file_path}")

        # Read the dataset
        data = pd.read_csv(csv_file_path)
        images = data['pixels']

        # Iterate through all image rows and save them
        for index, pixel_str in tqdm(enumerate(images), total=len(images), desc=f"Saving {datatype} images"):
            img = self.str_to_image(pixel_str)
            img.save(folder_path / f'{datatype}{index}.jpg', 'JPEG')

        print(f"Completed saving {datatype} images to {folder_path}.")
