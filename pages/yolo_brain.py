import torch
import streamlit as st
from PIL import Image
import os
import matplotlib.pyplot as plt
from torchvision.io import read_image
import torchvision.transforms as T
from PIL import Image
import requests
import numpy as np
import io
from PIL import ImageDraw, ImageFont

# ЗАГРУЗКА МОДЕЛИ
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# путь к папке с этим файлом
current_dir = os.path.dirname(os.path.abspath(__file__))
# путь к папке c yolov5:
project_root = os.path.abspath(os.path.join(current_dir, '..', 'yolov5'))
# путь к весам
weights_path = os.path.join(current_dir, '..', 'model', 'best (1).pt')

# try:
#     model = torch.hub.load(
#         repo_or_dir = project_root,
#         model = 'custom',
#         path=weights_path, 
#         source='local',
#         device=device
#     )
#     model.conf = 0.3
#     model.eval()

# except Exception as e:
#     st.error(f"Ошибка загрузки модели: {e}")
#     st.stop() # Остановка Streamlit приложения

model = torch.hub.load(
    repo_or_dir='ultralytics/yolov5',  # Официальный репозиторий YOLOv5 на GitHub
    model='custom',                    # Используем кастомную модель
    path=weights_path,   # Путь к весам относительно корня проекта
    trust_repo=True                    # Доверяем репозиторию (необходимо для кастомных моделей)
)

# ПАРАМЕТРЫ МОДЕЛИ        
def display_model_parameters():
    st.subheader("Параметры модели")
    st.write(f"Название модели: YOLOv5")
    st.write(f"Количество эпох: 50")
    st.write(f"Количество батчей: 25")
    st.write(f'Количество слоев: 157')
    st.write(f'Всего параметров - 1761871') #Веса и смещения
    st.write(f'mAP50: 0.936')
    st.write(f'mAP50-95: 0.63')

# Функция для детекции объектов
def detect_objects(image, model):
    results = model(image)
    return results

# Функция для отображения результата детекции
def display_results(image, results):
    # Преобразуем изображение в формат PIL
    image = Image.fromarray(image)
    # Рисуем рамки вокруг объектов
    for box in results.xyxy[0]:
        x1, y1, x2, y2 = box[:4].int().tolist()
        confidence = box[4].item()
        label = results.names[int(box[5])]
        # Рисуем рамку вокруг объекта
        draw = ImageDraw.Draw(image)
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=4)
        # Выводим название объекта и уровень уверенности
        draw.text((x1, y1 - 15), f"{label} ({confidence:.2f})", fill="red")
    # Отображаем результат
    return image

# Загрузка изображения
uploaded_files = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
# Ввод URL картинки
image_url = st.text_input("Введите URL картинки")

images = []
if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        try:
            image = Image.open(uploaded_file)
            image = np.array(image)
            results = detect_objects(image, model)
            detected_image = display_results(image, results)
            images.append(detected_image)
        except Exception as e:
            st.error(f"Ошибка обработки загруженного файла: {e}")

if image_url:
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Проверка на ошибки HTTP

        image = Image.open(response.raw)
        image = np.array(image)
        results = detect_objects(image, model)
        detected_image = display_results(image, results)
        images.append(detected_image)

    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка загрузки изображения по URL: {e}")

if images:
    #Отображение результатов
    st.image(images, width=None, caption=None)
    display_model_parameters()