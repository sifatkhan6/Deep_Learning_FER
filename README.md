# Facial Expression Recognition Using Attentional Convolutional Network

Research Paper, [ACN](https://arxiv.org/abs/1902.01019).

## Datasets
This implementation uses the following datasets:
- [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

## Prerequisites
Make sure you have the following libraries installed:
- PyTorch >= 1.1.0
- torchvision == 0.5.0
- OpenCV
- tqdm
- Pillow (PIL)

## Repository Structure
This repository is organized as follows:
- [`main`](/main.py): Contains setup for the dataset and training loop.
- [`visualize`](/visualize.py): Includes source code for evaluating the model on test data and real-time testing using a webcam.
- [`deep_emotion`](/deep_model.py): Defines the model class.
- [`data_loaders`](/data_loaders.py): Contains the dataset class.
- [`generate_data`](/generate_data.py): Setting up dataset
