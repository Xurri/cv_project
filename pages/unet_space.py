from pathlib import Path
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from model.unet.model_unet_class import UNet


# Инициализируем модель и загружаем веса
# --------------------------------------

# Определяем путь к директории скрипта
script_path = Path(__file__).resolve()
script_dir = script_path.parent
# Строим путь к весам модели
weights_path = script_dir.parent / 'model' / 'unet' / 'model_epoch_31.pth'

# Определяем device для выполнения предсказаний
# DEVICE = torch.device(
#     'cuda' if torch.cuda.is_available() else
#     'mps' if torch.backends.mps.is_available() else
#     'cpu'
# )
DEVICE = torch.device('cpu')

@st.cache_resource
def load_model():
    model = UNet(n_class=1)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model

model = load_model()
model = model.to(DEVICE)

# Определяем трансформации
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor()
])

# Титул страницы
st.title("UNet Segmentation App")
# описательная часть
description = """
    ### Архитектура U-Net для сегментации изображений
    
    **U-Net** представляет собой сверточную нейронную сеть, оптимизированную для точной сегментации изображений размером **320x320 пикселей**. Архитектура состоит из двух основных частей: **энкодера** и **декодера**.
    
    **Энкодер** включает **5 блоков**, каждый из которых содержит:
    - **Два сверточных слоя** (всего **10 сверточных слоёв**)
    - **Один слой max pooling** (за исключением последнего блока)
    
    Эти блоки последовательно уменьшают пространственное разрешение изображения, одновременно увеличивая глубину признаков, что позволяет эффективно извлекать высокоуровневые особенности из входных данных.
    
    **Декодер** состоит из **4 блоков**, каждый из которых включает:
    - **Один транспонированный сверточный слой** для увеличения разрешения
    - **Два обычных сверточных слоя** (всего **8 сверточных слоёв**)
    
    Кроме того, декодер использует **skip connections** для объединения признаков из соответствующих слоёв энкодера, что обеспечивает сохранение пространственных деталей и повышает точность сегментации.
    
    В завершение, **выходной слой** представляет собой **один сверточный слой**, который генерирует карту сегментации с необходимым числом классов. В общей сложности, архитектура модели включает **19 сверточных слоёв**.
    """

# Отображение описания модели с использованием Markdown
st.markdown(description)

# Первый блок с загрузкой картинок пользователем
# ----------------------------------------------
st.header('Предсказание UNet для загруженных пользователем изображений')

uploaded_files = st.file_uploader(
    "Выберите одно или несколько изображений",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
    )
if uploaded_files is not None:
    if st.button('Получить сегментации для файлов'):
        for uploaded_file in uploaded_files:
            # Открываем изображение
            image = Image.open(uploaded_file).convert('RGB')

            # Применяем трансформации
            input_image = transform(image).unsqueeze(0).to(DEVICE)

            # Получаем предсказание
            with torch.inference_mode():
                output = model(input_image)
            pred_mask = torch.sigmoid(output)
            pred_mask = (pred_mask > 0.5).float()

            # Преобразуем для отображения
            pred_mask_np = pred_mask.squeeze().cpu().numpy()

            # Отображаем предсказанную маску
            st.subheader(f'Сегментация для: {uploaded_file.name}')
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption='Original image', use_column_width=True)
            with col2:
                st.image(pred_mask_np, caption='Predicted mask', use_column_width=True)


# Второй блок с демонстрацией перфоманса модели
# ----------------------------------------------

st.header('Демонстрация работы UNet на данных, имеющих разметку')
if st.button('Посмотреть сравнительные результаты!'):
    images_paths = [
        script_dir.parent / 'images' / 'unet' / 'img' / 'img_1.jpg',
        script_dir.parent / 'images' / 'unet' / 'img' / 'img_2.jpg',
        script_dir.parent / 'images' / 'unet' / 'img' / 'img_3.jpg'
    ]
    masks_paths = [
        script_dir.parent / 'images' / 'unet' / 'mask' / 'mask_1.jpg',
        script_dir.parent / 'images' / 'unet' / 'mask' / 'mask_2.jpg',
        script_dir.parent / 'images' / 'unet' / 'mask' / 'mask_3.jpg'
    ]

    for i in range(3):
        # Загрузка изображения и маски
        image = Image.open(images_paths[i]).convert('RGB')
        true_mask = Image.open(masks_paths[i]).convert('L')

        # Применяем трансформации
        input_image = transform(image).unsqueeze(0).to(DEVICE)
        true_mask_tensor = transform(true_mask).squeeze(0)  # Размер (H, W)

        # Получаем предсказание модели
        with torch.inference_mode():
            output = model(input_image)
        pred_mask = torch.sigmoid(output)
        pred_mask = (pred_mask > 0.5).float()

        # Преобразуем тензоры в numpy массивы для отображения
        img_np = input_image.squeeze(0).cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)

        pred_mask_np = pred_mask.squeeze().cpu().numpy()
        true_mask_np = true_mask_tensor.cpu().numpy()

        # Отображение результатов
        st.subheader(f'Демо изображение № {i+1}')
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(img_np, caption='Original image', use_column_width=True)
        with col2:
            st.image(pred_mask_np, caption='Predicted mask', use_column_width=True)
        with col3:
            st.image(true_mask_np, caption='True mask', use_column_width=True)
