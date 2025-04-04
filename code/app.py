import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO 
from torchvision import transforms
import torch
from torchvision.models import vgg13, VGG13_Weights
import torch.nn.functional as Fun
import torch.nn as nn

CHANNEL_MEAN = [0.485, 0.456, 0.406]
CHANNEL_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToTensor(),  # Преобразует в тензор и нормализует в [0, 1]
    transforms.Resize((256, 256)),  # Изменяет размер
    transforms.Normalize(mean=CHANNEL_MEAN, std=CHANNEL_STD),  # Нормализует
])
white_thresholds = {
        'red': [100, 255],
        'green': [155, 255],
        'blue': [155, 255]
    }

transform = transforms.Compose([
    transforms.ToTensor(),  # Преобразует в тензор и нормализует в [0, 1]
    transforms.Resize((256, 256)),  # Изменяет размер
    transforms.Normalize(mean=CHANNEL_MEAN, std=CHANNEL_STD),  # Нормализует
])


class VGG13Encoder(torch.nn.Module):
    def __init__(self, num_blocks, weights=VGG13_Weights.DEFAULT):
        super().__init__()
        self.num_blocks = num_blocks
        feature_extractor = vgg13(weights=weights).features
        self.blocks = torch.nn.ModuleList()
        for idx in range(self.num_blocks):
            self.blocks.append(nn.Sequential(
                feature_extractor[idx*5:idx*5 + 4])
            )
    def forward(self, X):
        activations = []
        for idx, block in enumerate(self.blocks):
            X = block(X)
            activations.append(X)
            X = torch.functional.F.max_pool2d(X, kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        return activations

class DecoderBlock(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.upconv = torch.nn.Conv2d(
            in_channels=out_channels * 2, out_channels=out_channels,
            kernel_size=3, padding=1, dilation=1
        )
        self.conv1 = torch.nn.Conv2d(
            in_channels=out_channels * 2, out_channels=out_channels,
            kernel_size=3, padding=1, dilation=1
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, padding=1, dilation=1
        )
        self.relu = torch.nn.ReLU()
    def forward(self, down, left):
        x = Fun.interpolate(down, scale_factor=2, mode='nearest')
        x = self.upconv(x)
        x = torch.cat([x, left], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class Decoder(torch.nn.Module):
    def __init__(self, num_filters, num_blocks):
        super().__init__()
        self.blocks = torch.nn.ModuleList()
        for idx in range(num_blocks):
            self.blocks.insert(0, DecoderBlock(num_filters * 2 ** idx))
    def forward(self, acts):
        up = acts[-1]
        for block, left in zip(self.blocks, acts[-2::-1]):
            up = block(up, left)
        return up

class UNet(torch.nn.Module):
    def __init__(self, num_classes=1, num_blocks=4):
        super().__init__()
        self.encoder = VGG13Encoder(num_blocks)
        self.decoder = Decoder(64, num_blocks-1)
        self.final = nn.Conv2d(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=1
        )
    def forward(self, x):
        acts = self.encoder.forward(x)
        x = self.decoder.forward(acts)
        x = self.final(x)
        return x
    
def countur(mask, min_area, max_area, ellipse=False):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area] #and cv2.contourArea(cnt) < max_area]
    if ellipse:
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    (center, axes, angle) = ellipse
                    major_axis = max(axes)
                    minor_axis = min(axes)
                    aspect_ratio = minor_axis / major_axis

                    if 0.5 < aspect_ratio < 1.5:
                        filtered_contours.append(contour)
          
    mask_combined = np.zeros_like(mask)
    for contour in filtered_contours:
        if cv2.contourArea(contour) < max_area:
            m = np.zeros_like(mask)
            cv2.drawContours(m, [contour], -1, 255, thickness=cv2.FILLED)
            mask_combined = cv2.bitwise_or(mask_combined, m)
        else:
            return mask

    return mask_combined


def rgb_binarization(img, thresholds):
    combined_mask = np.ones(img.shape[:2], dtype=np.uint8)
    
    for channel, (lower, upper) in thresholds.items():
        # Выбор канала
        if channel == 'red':
            ch = img[:, :, 0]
        elif channel == 'green':
            ch = img[:, :, 1]
        elif channel == 'blue':
            ch = img[:, :, 2]

        mask = cv2.inRange(ch, lower, upper)
        combined_mask = cv2.bitwise_and(combined_mask, mask)
    return combined_mask

def eros_dilat(image, er_kernel_size, dilat_kernel_size):
    kernel = np.ones(er_kernel_size, np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    kernel = np.ones(dilat_kernel_size, np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    return image

def normalized(image):
    normalized_tensor = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Если нужно преобразовать в диапазон [0, 255] (для uint8)
    normalized_tensor_uint8 = (normalized_tensor * 255).astype(np.uint8)
    return normalized_tensor_uint8

def white(image):
        carousel = [image]
        captions = ["Original image"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet(num_classes=1, num_blocks=3)
        model.load_state_dict(torch.load("./best_model_epoch_9.pth", map_location=torch.device('cpu')))
        model.eval()

        image = transform(image)
        image = image.to(device)
        image = image.unsqueeze(0)
        logits = model(image).squeeze(0).detach().cpu().numpy().squeeze(0)
        carousel.append(normalized(logits))
        captions.append("Segmentation probability distribution")

        _, binary = cv2.threshold(logits, 11, 255, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)
        carousel.append(normalized(binary))
        captions.append("Mask")

        all_eggs = countur(binary, 170, 2000, ellipse=False)
        all_eggs_dil = eros_dilat(all_eggs, (1, 1), (10, 10))

        image = carousel[0]
        all_eggs_rgb = cv2.bitwise_and(image, image, mask=all_eggs_dil)
        carousel.append(all_eggs_rgb)
        captions.append("Image under the mask")
        
        wight_mask = rgb_binarization(all_eggs_rgb, white_thresholds)
        wight_mask = countur(wight_mask, 600, 2000, ellipse=False)
        carousel.append(normalized(wight_mask))
        captions.append("Mask for white eggs")

        
        wight_eggs = cv2.bitwise_and(image, image, mask=wight_mask)
        carousel.append(wight_eggs)
        captions.append("The final image under the mask")

        return carousel, captions

def red(image):
        carousel = [image]
        captions = ["Original image"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet(num_classes=1, num_blocks=3)
        model.load_state_dict(torch.load("./best_model_epoch_9 (2).pth", map_location=torch.device('cpu')))
        model.eval()

        image = transform(image)
        image = image.to(device)
        image = image.unsqueeze(0)
        logits = model(image).squeeze(0).detach().cpu().numpy().squeeze(0)
        carousel.append(normalized(logits))
        captions.append("Segmentation probability distribution")

        _, binary = cv2.threshold(logits, -6.3, 255, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)
        carousel.append(normalized(binary))
        captions.append("Mask")

        binary_clean = eros_dilat(binary, (1, 1), (2, 2))
        binary_full = countur(binary_clean, 500, 9900, ellipse=False)
        binary_full = cv2.morphologyEx(binary_full, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        carousel.append(normalized(binary_full))
        captions.append("Mask after noise removal and void filling")

        image = carousel[0]
        red_eggs = cv2.bitwise_and(image, image, mask=binary_full)
        carousel.append(red_eggs)
        captions.append("Final image under the mask")

        return carousel, captions

def process_yolo(image):
    temp_path = "temp_image.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    model = YOLO("./best.pt")
    results = model.predict(temp_path, conf=0.5)
    os.remove(temp_path)
    result_image = image.copy()
    
    class_colors = {
        "white": (200, 200, 200),
        "brown": (50, 50, 50)
    }
    
    # Отрисовываем bounding boxes с подписями
    for box in results[0].boxes:
        # Координаты bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # Класс и уверенность
        class_id = int(box.cls)
        class_name = results[0].names[class_id]  # Получаем имя класса из модели
        
        # Используем "brown" как fallback, если класс неизвестен
        confidence = float(box.conf)
        color = class_colors[class_name]
        
        # Рисуем bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        # Текст подписи
        label = f"{class_name} {confidence:.2f}"
        
        # Размер текста и отступы
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX , 0.3, 0)
        
        # Рисуем подложку для текста
        cv2.rectangle(result_image, 
                      (x1, y1 - text_height - 10), 
                      (x1 + text_width, y1), 
                      color, -1)
        
        # Рисуем текст
        cv2.putText(result_image, label, 
                    (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, 
                    (255, 255, 255), 1)
    
    return [result_image], ["Detection resutl"]

def segment_image(image, mode):
    if mode == "Red eggs":
        return red(image)
    elif mode == "White eggs":
        return white(image)

def display_carousel(images, captions, key_prefix):
    if f"current_index_{key_prefix}" not in st.session_state:
        st.session_state[f"current_index_{key_prefix}"] = 0

    st.subheader("Image segmentation process")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Prev", key=f"prev_button_{key_prefix}"):
            st.session_state[f"current_index_{key_prefix}"] = max(0, st.session_state[f"current_index_{key_prefix}"] - 1)
    with col2:
        if st.button("Next", key=f"next_button_{key_prefix}"):
            st.session_state[f"current_index_{key_prefix}"] = min(len(images) - 1, st.session_state[f"current_index_{key_prefix}"] + 1)

    st.image(images[st.session_state[f"current_index_{key_prefix}"]], 
             caption=captions[st.session_state[f"current_index_{key_prefix}"]], 
             width=500)

def clear_session_state():
    keys_to_keep = ['carosel', 'captions', 'current_index']
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]

def main():
    st.title("Egg segmentation")
    st.sidebar.title("Menu")
    if 'prev_model' not in st.session_state:
        st.session_state.prev_model = None
    model_type = st.sidebar.radio("Choose model", ["YOLO", "U-NET"])
    if model_type != st.session_state.prev_model:
        clear_session_state()
        st.session_state.prev_model = model_type

    if model_type == "U-NET":
        mode = st.sidebar.radio("Choose mode", ["Red eggs", "White eggs"])
    else:
        mode = None

    uploaded_file = st.sidebar.file_uploader("Upload your image", 
                                           type=["jpg", "jpeg", "png"])

    folder_path = "./data"
    if os.path.exists(folder_path):
        sample_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        sample_file = st.sidebar.selectbox("Or select sample image", 
                                         [None] + sample_files)
    else:
        sample_files = []
        sample_file = None

    if uploaded_file is not None or sample_file is not None:
        if st.sidebar.button("Segment"):
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                image = np.array(image)
            else:
                image_path = os.path.join(folder_path, sample_file)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.resize(image, (256, 256))

            if model_type == "U-NET":
                carosel, captions = segment_image(image, mode)
            else:
                carosel, captions = process_yolo(image)
            
            st.session_state.carosel = carosel
            st.session_state.captions = captions
            st.session_state.current_index = 0

        if "carosel" in st.session_state and "captions" in st.session_state:
            display_carousel(st.session_state.carosel, st.session_state.captions, key_prefix="single")

if __name__ == "__main__":
    main()