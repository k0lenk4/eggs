from .model import UNet
from .marph import countur, rgb_binarization, eros_dilat, normalized
from torchvision import transforms
import torch
import numpy as np
import cv2

CHANNEL_MEAN = [0.485, 0.456, 0.406]
CHANNEL_STD = [0.229, 0.224, 0.225]

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