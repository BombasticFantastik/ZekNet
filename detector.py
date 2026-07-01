import os
import logging
import cv2
import numpy as np
from insightface.model_zoo import get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def init_detector(model_path, target_size=2048):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели '{model_path}' не найден.")
        
    logging.info(f"Инициализация модели: {model_path}")
    detector = get_model(model_path, providers=["CPUExecutionProvider"])
    
    detector.prepare(
        ctx_id=-1,
        input_size=(target_size, target_size),
        nms_thresh=0.3
    )
    return detector


def detect_faces(image_path, detector, conf_thresh=0.45):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Изображение '{image_path}' не найдено.")

    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError(f"Не удалось открыть изображение '{image_path}'.")

    orig_h, orig_w = orig_img.shape[:2]

    # Рамка, чтобы детектор не терял лица у самых краев кадра
    BORDER = 150
    padded_img = cv2.copyMakeBorder(
        orig_img, BORDER, BORDER, BORDER, BORDER,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    logging.info(f"Детекция на {image_path} ({orig_w}x{orig_h})")
    
    bboxes, kpss = detector.detect(padded_img, max_num=0, metric="default")

    if bboxes is None or len(bboxes) == 0:
        logging.warning("Лица не найдены.")
        return []

    scores = bboxes[:, 4]
    rects = bboxes[:, :4]
    faces = []

    for i, bbox in enumerate(rects):
        score = float(scores[i])

        if score < conf_thresh:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        x1 -= BORDER
        y1 -= BORDER
        x2 -= BORDER
        y2 -= BORDER
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        # Перенос координат ключевых точек (keypoints) обратно к оригиналу
        # Из каждой точки kps вычитаем только BORDER, так как они были найдены на padded_img
        shifted_kps = kpss[i] - np.array([BORDER, BORDER])

        faces.append({
            "score": score,
            "bbox": (x1, y1, x2, y2),  # Чистые координаты лица на исходном фото
            "kps": shifted_kps         # Чистые координаты точек (глаза, нос, углы рта)
        })

    logging.info(f"Детекция завершена. Найдено лиц: {len(faces)} (из {len(bboxes)} гипотез)")
    return faces


if __name__ == "__main__":
    MODEL_FILE = "scrfd_500m_bnkps.onnx"
    INPUT_IMAGE = "faces/group_photo.png"
    
    try:
        face_detector = init_detector(MODEL_FILE, target_size=2048)
        
        detected_faces = detect_faces(
            image_path=INPUT_IMAGE,
            detector=face_detector,
            conf_thresh=0.45
        )
        
        for idx, face in enumerate(detected_faces):
            print(f"Лицо #{idx + 1}: Score: {face['score']:.2f} | BBox (x1,y1,x2,y2): {face['bbox']}")
            
    except Exception as e:
        logging.error(f"Критический сбой скрипта: {e}", exc_info=True)