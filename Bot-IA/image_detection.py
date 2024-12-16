import cv2
import io
import numpy as np
from PIL import Image, ImageDraw

YOLO_CONFIG = "yolov3.cfg"
YOLO_WEIGHTS = "yolov3.weights"
YOLO_CLASSES = "coco.names"

net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
with open(YOLO_CLASSES, "r") as f:
    classes = [line.strip() for line in f.readlines()]

def detectar_objetos(img_data, objeto):
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    img_np = np.array(img)

    height, width, _ = img_np.shape

    blob = cv2.dnn.blobFromImage(img_np, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    detections = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    procesada = False
    if len(indexes) > 0:
        for i in indexes.flatten():
            if classes[class_ids[i]] == objeto:
                x, y, w, h = boxes[i]
                draw = ImageDraw.Draw(img)
                draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
                draw.text((x, y - 10), f"{objeto}: {confidences[i]:.2f}", fill="red")
                procesada = True

    return img, procesada
