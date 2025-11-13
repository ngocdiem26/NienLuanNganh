import os
import cv2
import numpy as np
import base64
import re
import threading
from ultralytics import YOLO
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# === CẤU HÌNH ===
yolo_model_path = "yolo_lp_train2/weights/best.pt"
input_folder = "input_images"
output_folder = "output_gemini"

os.makedirs(output_folder, exist_ok=True)

# ⚠️ Thay bằng API key thật của bạn
os.environ["GOOGLE_API_KEY"] = "AIzaSyC2kZozBC8TiUoml-QGQs9kgHHkqLdeLl4"

# === LOAD MODEL (KHÔNG fuse) ===
model = YOLO(yolo_model_path)
if hasattr(model, "model"):
    model.model.fuse = lambda *args, **kwargs: model.model
gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

# === Mapping ký tự thường gặp trong biển số VN ===
dict_char_to_int = {'O': '0', 'I': '1', 'B': '3', 'A': '4', 'G': '6', 'S': '5', 'L': '1', 'Z': '2', 'T': '1'}

def format_license_vn(text):
    text = text.upper()
    formatted = ''
    for i, ch in enumerate(text):
        if i != 2 and i != 3 and ch in dict_char_to_int:
            formatted += dict_char_to_int[ch]
        else:
            formatted += ch
    return formatted

# === HÀM HỖ TRỢ GEMINI ===
def encode_image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")

def gemini_read_plate(base64_image, filename):
    try:
        message = HumanMessage(
            content=[
                {"type": "text", "text": (
                    "You are a vision model. Please analyze this image of a Vietnamese vehicle number plate "
                    "and return ONLY the text you see (the plate number). Do not explain anything."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )
        response = gemini.invoke([message])
        text = response.content.strip().replace(" ", "").replace("\n", "")
        return format_license_vn(text)
    except Exception as e:
        print(f"[Gemini Error] {filename}:", e)
        return ""

# === GHI LOG ===
log_path = os.path.join(output_folder, "gemini_results.txt")
log_file = open(log_path, "w", encoding="utf-8")
lock = threading.Lock()

def process_image(filename):
    image_path = os.path.join(input_folder, filename)
    img = cv2.imread(image_path)
    if img is None:
        return

    results = model(img)
    found = False

    for result in results:
        for i, box in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            base64_img = encode_image_to_base64(crop)
            plate_text = gemini_read_plate(base64_img, filename)

            if plate_text:
                found = True
                cv2.putText(img, plate_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)

                with lock:
                    print(f"[✔] {filename} → {plate_text}")
                    log_file.write(f"{filename}: {plate_text}\n")

    if not found:
        with lock:
            print(f"[✖] {filename} → Không nhận diện được.")
            log_file.write(f"{filename}: [KHÔNG NHẬN RA]\n")

    output_path = os.path.join(output_folder, f"result_{filename}")
    cv2.imwrite(output_path, img)

# === XỬ LÝ TOÀN BỘ ẢNH (đa luồng) ===
threads = []
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        t = threading.Thread(target=process_image, args=(filename,))
        t.start()
        threads.append(t)

for t in threads:
    t.join()

log_file.close()
print(f"✅ Đã lưu kết quả tại: {output_folder}")
print(f"📄 File log: {log_path}")
