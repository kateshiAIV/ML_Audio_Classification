import os
import cv2
import matplotlib.pyplot as plt
from transformers import ViTImageProcessor
import numpy as np

# Путь к данным
data_dir = "Dataset/train"

# Классы (папки)
classes = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

images = []
labels = []

# Читаем изображения
for label, cls in enumerate(classes):
    cls_dir = os.path.join(data_dir, cls)
    for file in os.listdir(cls_dir):
        file_path = os.path.join(cls_dir, file)
        img = cv2.imread(file_path)
        if img is None:
            continue  # пропускаем битые файлы
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # переводим в RGB
        img = cv2.resize(img, (224, 224))  # под ViT обычно 224x224
        images.append(img)
        labels.append(label)

# Преобразуем в numpy
images = np.array(images)
labels = np.array(labels)

print("Размер выборки:", images.shape, labels.shape)

# ==============================
# Здесь можно вставить свой ML алгоритм:
# - разделение на train/test
# - обучение модели (например, ViT, CNN, SVM и т.д.)
# - оценка точности
# ==============================

# Пример визуализации
plt.figure(figsize=(10, 5))
for i in range(7):
    plt.subplot(1, 7, i+1)
    plt.imshow(images[i])
    plt.title(classes[labels[i]])
    plt.axis("off")
plt.show()


