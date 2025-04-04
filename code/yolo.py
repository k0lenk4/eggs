import cv2
from ultralytics import YOLO 
import os
import numpy as np

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