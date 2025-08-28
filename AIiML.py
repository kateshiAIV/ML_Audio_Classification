import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Путь к данным
data_dir = "Dataset/train"

# Классы (папки)
classes = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

images = []
labels = []

# Читаем изображения как есть
for label, cls in enumerate(classes):
    cls_dir = os.path.join(data_dir, cls)
    for file in os.listdir(cls_dir):
        file_path = os.path.join(cls_dir, file)
        img = cv2.imread(file_path)  # читаем без resize и преобразований
        if img is None:
            continue  # пропускаем битые файлы
        images.append(img)
        labels.append(label)

# Преобразуем в numpy
images = np.array(images)
labels = np.array(labels)

print("Размер выборки:", images.shape, labels.shape)

# ==============================
# Здесь можно вставить свой ML алгоритм:
# - разделение на train/test
# - обучение модели (например, CNN, SVM и т.д.)
# - оценка точности
# ==============================

import random

plt.figure(figsize=(15, 5))

for i, cls in enumerate(classes):
    # индексы всех картинок данной категории
    cls_indices = np.where(labels == i)[0]
    # выбираем случайный индекс
    idx = random.choice(cls_indices)
    # берём картинку
    img = images[idx]

    plt.subplot(1, 7, i+1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(cls)
    plt.axis("off")

plt.show()

