import cv2
import numpy as np


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