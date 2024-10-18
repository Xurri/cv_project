import torch
from PIL import Image, ImageFilter
import requests
import os
from io import BytesIO
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.io import read_image
from IPython.display import display
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', 'yolov5'))
weights_path = os.path.join(current_dir, '..', 'model', 'yolo_face_best.pt')
image_path = os.path.join(current_dir, '..', 'images', 'yolo_face_results.png')

print("Project root:", project_root)

model = torch.hub.load(
    # будем работать с локальной моделью в текущей папке
    repo_or_dir = project_root,
    # непредобученная – будем подставлять свои веса
    model = 'custom',
    # путь к нашим весам
    path = weights_path,
    # откуда берем модель – наша локальная
    source='local'
    )

# Начиная с какой вероятности отрисовывать детекции
model.conf = 0.40
# Читаем картинку
#img = T.ToPILImage()(read_image('/home/lbeno/ds_Elbrus_/phase_2_projects/cv_project/yolov5/data/images/val/0af4a09ce6783cb6.jpg'))
st.markdown("**1. Ссылка для обработки изображения**")
url = st.text_input('Введите URL изображения:') #'https://www.socialnicole.com/wp-content/uploads/2015/02/youngsters.jpg'
if url is not None:
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img = img.resize((640, 640))
        model.eval()
        with torch.inference_mode():
            results = model(img)
        # results.show()  # or .show(), .save(), .crop(), .pandas(), render(), etc




        detections = results.xyxy[0]  # Detections for the first image

        # Loop over each detection and apply blur to the detected area
        for det in detections:
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Define the bounding box
            box = (x1, y1, x2, y2)

            # Crop the detected area
            region = img.crop(box)

            # Apply Gaussian Blur to the cropped area
            blurred_region = region.filter(ImageFilter.GaussianBlur(radius=15))

            # Paste the blurred area back into the image
            img.paste(blurred_region, box)

        # Display the image with blurred detections
        # At the end of your script
        #img.save('output_blurred.jpg')
        st.image(img, caption="Вот и вся история")
    except:
        st.markdown("**Формат изображения не подходит, смените на другой**")
else:
    st.markdown("**Введите URL изображения:**")

st.markdown("**2. Если у вас много фоток, кидайте сюда**")
uploaded_files = st.file_uploader("pics pls here", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            img = Image.open(uploaded_file)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Запуск модели
            model.eval()
            with torch.inference_mode():
                results = model(img)

            # Получение детекций (bounding boxes)
            detections = results.xyxy[0]

            # Проверяем, есть ли детекции
            if len(detections) == 0:
                st.write(f"Объекты не обнаружены на изображении {uploaded_file.name}.")
            else:
                # Применение размытия к обнаруженным областям
                for det in detections:
                    x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    box = (x1, y1, x2, y2)
                    region = img.crop(box)
                    blurred_region = region.filter(ImageFilter.GaussianBlur(radius=15))
                    img.paste(blurred_region, box)

                # Отображаем обработанное изображение
                st.image(img, caption=f'Обработанное изображение: {uploaded_file.name}', use_column_width=True)
        except Exception as e:
            st.error(f"Произошла ошибка при обработке файла {uploaded_file.name}: {e}")
else:
    st.write("Пожалуйста, загрузите одно или несколько изображений для обработки.")

st.markdown("**3. Здесь вы можете лицезреть, ненавистные всем графики и другую информацию**")
st.markdown("3.1 В этом датасете было 13386 трейновых файлов и 3387 валидационных файлов")
st.markdown("3.2 Модель обучалась локально и всего на 4 эпохах")
st.markdown("3.3 Батч сайз составлял 45 файлов единоразово, с запасом памяти на видеокарте")
st.markdown("3.4 На графиках ниже, мы видим положительную динамику обучения модели, как по функции потерь, так и по метрикам.")
st.image(image_path, caption="Как то так, как то так")