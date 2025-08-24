import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Ukrycie ostrzeżeń CUDA i innych INFO/WARNING
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Wyłączenie GPU, działanie tylko na CPU
import cv2
import matplotlib.pyplot as plt
from transformers import ViTImageProcessor
import numpy as np
image = cv2.imread("test.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.axis('off')
plt.title("Oryginalny obraz")
plt.show()
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
inputs = processor(images=image, return_tensors="tf")
