import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

class PlainDataset(Dataset):
    def __init__(self, csv_file, img_dir, datatype, transform=None):
        """
        PyTorch Dataset class for loading images and their labels.

        Args:
            csv_file (str): Path to the CSV file with image labels.
            img_dir (str): Directory containing the images.
            datatype (str): The data type ('train', 'val', or 'test').
            transform (torchvision.transforms.Compose, optional): Optional transformations to apply to images.
        """
        # Ensure the CSV file exists
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        self.csv_file = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.datatype = datatype
        self.transform = transform or transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        # Ensure the datatype is valid
        valid_datatypes = ['train', 'val', 'test']
        if self.datatype not in valid_datatypes:
            raise ValueError(f"Invalid datatype '{self.datatype}', expected one of {valid_datatypes}")

        self.labels = self.csv_file['emotion']

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        """Get image and corresponding label at the given index."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.img_dir, f"{self.datatype}{idx}.jpg")

        # Check if the image exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Load the image
        img = Image.open(img_path)

        # Get the label and convert it to tensor
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.long)

        # Apply transformation if available
        if self.transform:
            img = self.transform(img)

        return img, label


def display_sample(csv_file, img_dir, datatype, sample_number, transform=None):
    """
    Helper function to load a sample from the dataset and display the image and label.

    Args:
        csv_file (str): Path to the CSV file with image labels.
        img_dir (str): Directory containing the images.
        datatype (str): The data type ('train', 'val', 'test').
        sample_number (int): The sample index to load and display.
        transform (torchvision.transforms.Compose, optional): Transformations to apply to the image.
    """
    dataset = PlainDataset(csv_file=csv_file, img_dir=img_dir, datatype=datatype, transform=transform)

    # Retrieve the sample data
    img, label = dataset[sample_number]

    # Convert the image tensor to a numpy array for displaying
    img_numpy = img.numpy().transpose(1, 2, 0)  # Convert from CHW to HWC format for plt.imshow
    img_numpy = np.clip(img_numpy, 0, 1)  # Clip values to [0, 1] for display

    # Display the image
    plt.imshow(img_numpy)
    plt.title(f"Label: {label.item()}")
    plt.axis('off')
    plt.show()
