import os
import cv2
import numpy as np
import re
import easyocr
import time
from ultralytics import YOLO

# === CẤU HÌNH ===
yolo_model_path = "yolo_lp_train3/weights/best.pt"
output_folder = "output_realtime"
# debug_folder = "debug_steps"
os.makedirs(output_folder, exist_ok=True)
# os.makedirs(debug_folder, exist_ok=True)

# === LOAD MODEL ===
yolo_model = YOLO(yolo_model_path)
ocr_reader = easyocr.Reader(['en'], gpu=False)

# === DICTIONARY SỬA LỖI KÝ TỰ ===
dict_char_to_int = {'O': '0', 'I': '1', 'B': '3', 'A': '4', 'G': '6', 'S': '5', 'L': '1', 'Z': '2', 'T': '1'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'B', '4': 'A', '6': 'G', '5': 'S', '7': 'Z', '8': 'B', '9': 'O'}

def format_license_vn(text):
    text = text.upper()
    formatted = ''
    for i, ch in enumerate(text):
        if i == 2 and ch in dict_int_to_char:
            formatted += dict_int_to_char[ch]
        elif i != 2 and i != 3 and ch in dict_char_to_int:
            formatted += dict_char_to_int[ch]
        else:
            formatted += ch
    return formatted

def crop_tight_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return plate_img
    all_boxes = [cv2.boundingRect(c) for c in contours]
    xs = [x for (x, y, w, h) in all_boxes]
    ys = [y for (x, y, w, h) in all_boxes]
    ws = [w for (x, y, w, h) in all_boxes]
    hs = [h for (x, y, w, h) in all_boxes]
    x1 = max(min(xs) - 3, 0)
    y1 = max(min(ys) - 3, 0)
    x2 = min(max([xs[i] + ws[i] for i in range(len(ws))] + [plate_img.shape[1]]), plate_img.shape[1])
    y2 = min(max([ys[i] + hs[i] for i in range(len(hs))] + [plate_img.shape[0]]), plate_img.shape[0])
    return plate_img[y1:y2, x1:x2]

def recognize_plate(image, x1, y1, x2, y2):
    plate_img = image[y1:y2, x1:x2]
    if plate_img.size == 0:
        return ""

    plate_img = crop_tight_plate(plate_img)
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(equalized)
    blur = cv2.GaussianBlur(contrast, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 10)

    ocr_result = ocr_reader.readtext(binary, detail=1)
    if len(ocr_result) == 0:
        return ""

    lines = []
    for (bbox, text, conf) in ocr_result:
        (tl, tr, br, bl) = bbox
        center_y = (tl[1] + bl[1]) / 2
        added = False
        for line in lines:
            if abs(line["cy"] - center_y) < 25:
                line["items"].append((tl[0], text))
                added = True
                break
        if not added:
            lines.append({"cy": center_y, "items": [(tl[0], text)]})

    lines.sort(key=lambda l: l["cy"])
    texts = []

    for line in lines:
        sorted_items = sorted(line["items"], key=lambda x: x[0])
        raw = ''.join([t[1] for t in sorted_items])
        clean = re.sub(r'[^A-Za-z0-9]', '', raw)
        texts.append(clean)

    if len(texts) == 2:
        raw_text = texts[0] + '-' + texts[1]
    elif len(texts) == 1:
        raw_text = texts[0]
    else:
        raw_text = ''

    formatted_text = format_license_vn(raw_text)
    return formatted_text

# === LOG FILE ===
result_realtime = os.path.join(output_folder, "output_results.txt")
f = open(result_realtime, "w", encoding="utf-8")

# === MỞ CAMERA ===
cap = cv2.VideoCapture(0) 
print("[▶] Nhấn 'q' để thoát.")

from datetime import datetime

# Khởi tạo camera
cap = cv2.VideoCapture(0)
print("[▶] Nhấn 'q' để thoát | Nhấn 's' để lưu ảnh có biển số nhận diện được.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[!] Không đọc được khung hình.")
        break

    plate_text_found = None  # Mỗi frame reset

    results = yolo_model(frame)
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            plate_text = recognize_plate(frame, x1, y1, x2, y2)
            if plate_text:
                plate_text = plate_text.upper()  # CHUYỂN THÀNH IN HOA
                plate_text_found = plate_text
                print(f"[✔] Biển số: {plate_text}")
                cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

    cv2.imshow("License Plate Recognition - Press Q to quit | S to save", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and plate_text_found:
        # Chỉ lưu nếu có biển số nhận diện
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plate_{plate_text_found}_{timestamp}.jpg"
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, frame)
        print(f"[💾] Đã lưu ảnh: {output_path}")

cap.release()
cv2.destroyAllWindows()

f.close()

print(f"[📁] Đã lưu ảnh vào: {output_folder}")
print(f"[📄] File log kết quả: {result_realtime}")
