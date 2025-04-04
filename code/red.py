from model import UNet
from marph import countur, rgb_binarization, eros_dilat,  normalized
from torchvision import transforms
import torch
import numpy as np
import cv2

CHANNEL_MEAN = [0.485, 0.456, 0.406]
CHANNEL_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToTensor(),  # Преобразует в тензор и нормализует в [0, 1]
    transforms.Resize((256, 256)),  # Изменяет размер
    transforms.Normalize(mean=CHANNEL_MEAN, std=CHANNEL_STD),  # Нормализует
])

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

