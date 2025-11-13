import os
import cv2
import numpy as np
import re
import easyocr
from ultralytics import YOLO

# === CẤU HÌNH ===
yolo_model_path = "yolo_lp_train2/weights/best.pt"
input_folder = "input_images"
output_folder = "output_images"
debug_folder = "debug_steps"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(debug_folder, exist_ok=True)

# === LOAD MODEL ===
yolo_model = YOLO(yolo_model_path)
ocr_reader = easyocr.Reader(['en'], gpu=False)

# === Mapping ký tự thường gặp trong biển số VN ===
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

def keep_head_tail(text, total_len, head=2, tail=2):
    text = text.replace(" ", "")
    if len(text) <= total_len:
        return text
    return text[:head] + text[-tail:]

def recognize_plate(image, x1, y1, x2, y2, debug_prefix=None):
    plate_img = image[y1:y2, x1:x2]
    if plate_img.size == 0:
        return ""

    if debug_prefix:
        cv2.imwrite(os.path.join(debug_folder, f"{debug_prefix}_1_yolo_crop.jpg"), plate_img)

    plate_img = crop_tight_plate(plate_img)
    if debug_prefix:
        cv2.imwrite(os.path.join(debug_folder, f"{debug_prefix}_2_crop_tight.jpg"), plate_img)

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(equalized)
    blur = cv2.GaussianBlur(contrast, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 10)

    if debug_prefix:
        steps = [gray, equalized, contrast, blur, binary]
        labeled_steps = []
        for i, step in enumerate(steps):
            step_bgr = cv2.cvtColor(step, cv2.COLOR_GRAY2BGR)
            label = ["gray", "equalized", "clahe", "blur", "binary"][i]
            cv2.putText(step_bgr, label, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            labeled_steps.append(step_bgr)
        combined = cv2.hconcat(labeled_steps)
        cv2.imwrite(os.path.join(debug_folder, f"{debug_prefix}_4_compare.jpg"), combined)

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
                line["items"].append((tl[0], text, conf))
                added = True
                break
        if not added:
            lines.append({"cy": center_y, "items": [(tl[0], text, conf)]})

    lines.sort(key=lambda l: l["cy"])
    texts = []
    for line in lines:
        sorted_items = sorted(line["items"], key=lambda x: x[0])
        raw = ''.join([t[1] for t in sorted_items])
        clean = re.sub(r'[^A-Za-z0-9]', '', raw)
        texts.append(clean)

    if len(texts) == 1:
        final_text = texts[0].replace(" ", "")
    elif len(texts) == 2:
        line1 = keep_head_tail(texts[0], total_len=4, head=2, tail=2)
        line2 = texts[1].replace(" ", "")[:5]
        final_text = line1 + "-" + line2
    else:
        line1 = keep_head_tail(texts[0], total_len=4, head=2, tail=2)
        rest = "".join(texts[1:]).replace(" ", "")[:5]
        final_text = line1 + "-" + rest

    return format_license_vn(final_text)

# === GHI LOG KẾT QUẢ ===
result_log_path = os.path.join(output_folder, "output_results.txt")
f = open(result_log_path, "w", encoding="utf-8")

# === XỬ LÝ TẬP ẢNH ===
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    image_path = os.path.join(input_folder, filename)
    img = cv2.imread(image_path)
    if img is None:
        continue

    results = yolo_model(img)
    found = False

    for result in results:
        for box_id, box in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            debug_prefix = f"{os.path.splitext(filename)[0]}_{box_id}"
            plate_text = recognize_plate(img, x1, y1, x2, y2, debug_prefix)
            if plate_text:
                found = True
                print(f"[✔] {filename} → Biển số: {plate_text}")
                f.write(f"{filename}: {plate_text}\n")
                cv2.putText(img, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)

    if not found:
        print(f"[✖] {filename} → Không nhận diện được.")
        f.write(f"{filename}: [KHÔNG NHẬN RA]\n")

    output_path = os.path.join(output_folder, f"result_{filename}")
    cv2.imwrite(output_path, img)

f.close()
print(f" => Đã lưu kết quả tại: {output_folder}")
print(f" => Ảnh debug lưu tại: {debug_folder}")
print(f" => File log kết quả: {result_log_path}")
