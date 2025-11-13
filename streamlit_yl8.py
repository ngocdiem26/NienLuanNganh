
# import streamlit as st
# import numpy as np
# import cv2
# from PIL import Image
# import easyocr
# from ultralytics import YOLO
# import re
# import time
# import pandas as pd

# # === Khởi tạo SessionState để lưu nhiều kết quả ===
# if "results" not in st.session_state:
#     st.session_state.results = []

# # === Load model YOLOv10 và EasyOCR ===
# model = YOLO("yolo_lp_train2/weights/best.pt")
# ocr_reader = easyocr.Reader(['en'], gpu=False)

# # === Mapping ký tự thường gặp trong biển số VN ===
# dict_char_to_int = {'O': '0', 'I': '1', 'B': '3', 'A': '4', 'G': '6', 'S': '5', 'L': '1', 'Z': '2', 'T': '1'}
# dict_int_to_char = {'0': 'O', '1': 'I', '3': 'B', '4': 'A', '6': 'G', '5': 'S', '7': 'Z', '8': 'B', '9': 'O'}

# def format_license_vn(text):
#     text = text.upper()
#     formatted = ''
#     for i, ch in enumerate(text):
#         if i == 2 and ch in dict_int_to_char:
#             formatted += dict_int_to_char[ch]
#         elif i != 2 and i != 3 and ch in dict_char_to_int:
#             formatted += dict_char_to_int[ch]
#         else:
#             formatted += ch
#     return formatted

# def crop_tight_plate(plate_img):
#     gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return plate_img
#     all_boxes = [cv2.boundingRect(c) for c in contours]
#     xs = [x for (x, y, w, h) in all_boxes]
#     ys = [y for (x, y, w, h) in all_boxes]
#     ws = [w for (x, y, w, h) in all_boxes]
#     hs = [h for (x, y, w, h) in all_boxes]
#     x1 = max(min(xs) - 3, 0)
#     y1 = max(min(ys) - 3, 0)
#     x2 = min(max([xs[i] + ws[i] for i in range(len(ws))] + [plate_img.shape[1]]), plate_img.shape[1])
#     y2 = min(max([ys[i] + hs[i] for i in range(len(hs))] + [plate_img.shape[0]]), plate_img.shape[0])
#     return plate_img[y1:y2, x1:x2]

# # cắt ký tự ở giữa hay nhận diện sai
# def keep_head_tail(text, total_len, head=2, tail=2):
#     text = text.replace(" ", "")
#     if len(text) <= total_len:
#         return text
#     return text[:head] + text[-tail:]

# def recognize_plate(img, x1, y1, x2, y2):
#     plate_img = img[y1:y2, x1:x2]
#     if plate_img.size == 0:
#         return "", 0.0, []

#     plate_img = crop_tight_plate(plate_img)

#     gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
#     equalized = cv2.equalizeHist(gray)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     contrast = clahe.apply(equalized)
#     blur = cv2.GaussianBlur(contrast, (5, 5), 0)
#     binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                                    cv2.THRESH_BINARY_INV, 21, 10)

#     ocr_result = ocr_reader.readtext(binary, detail=1)
#     if len(ocr_result) == 0:
#         return "", 0.0, []

#     lines = []
#     char_list = []
#     for (bbox, text, conf) in ocr_result:
#         (tl, tr, br, bl) = bbox
#         center_y = (tl[1] + bl[1]) / 2
#         added = False
#         for line in lines:
#             if abs(line["cy"] - center_y) < 30:
#                 line["items"].append((tl[0], text, conf))
#                 added = True
#                 break
#         if not added:
#             lines.append({"cy": center_y, "items": [(tl[0], text, conf)]})

#     lines.sort(key=lambda l: l["cy"])
#     texts = []

#     for line in lines:
#         sorted_items = sorted(line["items"], key=lambda x: x[0])
#         for _, char, conf in sorted_items:
#             char_list.append((char, round(conf, 2)))
#         raw = ''.join([t[1] for t in sorted_items])
#         clean = re.sub(r'[^A-Za-z0-9]', '', raw)
#         texts.append(clean)



#     # if len(texts) == 2:
#     #     raw_text = texts[0] + '-' + texts[1]
#     # elif len(texts) == 1:
#     #     raw_text = texts[0]
#     # else:
#     #     raw_text = ''
#     if len(texts) == 1:
#         final_text = texts[0].replace(" ", "")
#     elif len(texts) == 2:
#     # Cắt line1 ở giữa: giữ 2 đầu (thường là mã tỉnh và loại xe)
#         line1 = keep_head_tail(texts[0], total_len=4, head=2, tail=2)
#         line2 = texts[1].replace(" ", "")[:5]  # giữ 5 ký tự đầu
#         final_text = line1 + "-" + line2
#     else:
#         line1 = keep_head_tail(texts[0], total_len=4, head=2, tail=2)
#         rest = "".join(texts[1:]).replace(" ", "")[:5]
#         final_text = line1 + "-" + rest



#     avg_conf = np.mean([conf for (_, _, conf) in ocr_result])
#     return format_license_vn(final_text), avg_conf, char_list

# # === Giao diện ===
# st.title("🚗 Nhận diện biển số xe bằng YOLOv8 + EasyOCR")

# uploaded_file = st.file_uploader("📤 Tải ảnh biển số xe", type=["jpg", "png", "jpeg"])
# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     image_np = np.array(image)
#     result_img = image_np.copy()
#     plate_results = []  # <-- lưu tất cả biển số từ ảnh hiện tại
#     start_time = time.time()
#     results = model(image_np)
#     inference_time = time.time() - start_time
#     for result in results:
#         for box in result.boxes.xyxy:
#             x1, y1, x2, y2 = map(int, box.cpu().numpy())
#             plate_text, confidence, char_info = recognize_plate(result_img, x1, y1, x2, y2)
#             if plate_text:
#                 text_color = (0, 255, 255)
#                 box_color = (233, 66, 62)
#                 cv2.putText(result_img, plate_text, (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 5)
#                 cv2.rectangle(result_img, (x1, y1), (x2, y2), box_color, 5)

#                 # Thêm từng kết quả vào list
#                 plate_results.append({
#                     "text": plate_text,
#                     "confidence": confidence,
#                     "chars": char_info,
#                     "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
#                 })

#     # ✅ Hiển thị ảnh sau xử lý
#     col1, col2 = st.columns(2)
#     with col1:
#         st.image(image, caption="Ảnh gốc", use_container_width=True)
#     with col2:
#         st.image(result_img, caption="Kết quả", channels="RGB", use_container_width=True)

#     # ✅ Ghi lại nhiều kết quả biển số
#     if plate_results:
#         st.markdown("### 🧾 Kết quả nhận diện")
#         for pr in plate_results:
#             st.session_state.results.append({
#                 "original": image.copy(),
#                 "result": result_img.copy(),
#                 **pr
#             })

#         # Hiển thị bảng kết quả của ảnh hiện tại
#         table_data_current = {
#             "Biển số": [p["text"] for p in plate_results],
#             "Thời điểm nhận diện": [p["timestamp"] for p in plate_results]
#         }
#         st.table(pd.DataFrame(table_data_current))

#     # ✅ Hiển thị toàn bộ lịch sử nhận diện
# if st.session_state.results:
#     st.markdown("🗂️ Lịch sử nhận diện")
#     all_data = {
#         "Biển số": [r["text"] for r in st.session_state.results],
#         "Thời điểm": [r["timestamp"] for r in st.session_state.results],
#     }
#     st.dataframe(pd.DataFrame(all_data), use_container_width=True)

# if st.button(" Xoá lịch sử"):
#     st.session_state.results = []
#     st.success("Đã xoá.")

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import easyocr
from ultralytics import YOLO
import re
import time
import pandas as pd

# === Khởi tạo SessionState để lưu nhiều kết quả ===
if "results" not in st.session_state:
    st.session_state.results = []
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

# === Load model YOLO và EasyOCR ===
model = YOLO("yolo_lp_train2/weights/best.pt")
ocr_reader = easyocr.Reader(['en'], gpu=False)

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

def recognize_plate(img, x1, y1, x2, y2):
    plate_img = img[y1:y2, x1:x2]
    if plate_img.size == 0:
        return "", 0.0, []

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
        return "", 0.0, []

    lines = []
    char_list = []
    for (bbox, text, conf) in ocr_result:
        (tl, tr, br, bl) = bbox
        center_y = (tl[1] + bl[1]) / 2
        added = False
        for line in lines:
            if abs(line["cy"] - center_y) < 30:
                line["items"].append((tl[0], text, conf))
                added = True
                break
        if not added:
            lines.append({"cy": center_y, "items": [(tl[0], text, conf)]})

    lines.sort(key=lambda l: l["cy"])
    texts = []

    for line in lines:
        sorted_items = sorted(line["items"], key=lambda x: x[0])
        for _, char, conf in sorted_items:
            char_list.append((char, round(conf, 2)))
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

    avg_conf = np.mean([conf for (_, _, conf) in ocr_result])
    return format_license_vn(final_text), avg_conf, char_list

# === Giao diện ===
st.set_page_config(layout="centered")
st.title("🚗 Hệ thống nhận diện biển số xe bằng YOLO + EasyOCR")

tab1, tab2 = st.tabs(["📸 Ảnh tĩnh", "📹 Webcam"])

# ====== TAB 1: Ảnh tĩnh ======
with tab1:
    uploaded_file = st.file_uploader("📤 Tải ảnh biển số xe", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        result_img = image_np.copy()
        plate_results = []
        start_time = time.time()
        results = model(image_np)
        inference_time = time.time() - start_time

        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                plate_text, confidence, char_info = recognize_plate(result_img, x1, y1, x2, y2)
                if plate_text:
                    text_color = (0, 255, 255)
                    box_color = (233, 66, 62)
                    cv2.putText(result_img, plate_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 5)
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), box_color, 5)
                    plate_results.append({
                        "text": plate_text,
                        "confidence": confidence,
                        "chars": char_info,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Ảnh gốc", use_container_width=True)
        with col2:
            st.image(result_img, caption="Kết quả", channels="RGB", use_container_width=True)

        if plate_results:
            st.markdown("### 🧾 Kết quả nhận diện")
            for pr in plate_results:
                st.session_state.results.append({
                    "original": image.copy(),
                    "result": result_img.copy(),
                    **pr
                })
            st.table(pd.DataFrame({
                "Biển số": [p["text"] for p in plate_results],
                "Thời điểm nhận diện": [p["timestamp"] for p in plate_results]
            }))

# ====== TAB 2: Webcam ======
with tab2:
    camera_placeholder = st.empty()

    if st.button("📷 Bật Camera"):
        st.session_state.camera_on = True

    if st.session_state.camera_on:
        cap = cv2.VideoCapture(0)
        st.info("⏱️ Đang nhận diện biển số xe")

        # Nút dừng realtime
        stop_btn = st.button("🛑 Dừng camera")

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                st.error("Không thể đọc từ webcam.")
                break

            result_img = frame.copy()
            plate_found = False
            results = model(frame)

            for result in results:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    plate_text, confidence, char_info = recognize_plate(result_img, x1, y1, x2, y2)
                    if plate_text:
                        plate_found = True
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        text_color = (0, 255, 255)
                        box_color = (233, 66, 62)
                        cv2.putText(result_img, plate_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)
                        cv2.rectangle(result_img, (x1, y1), (x2, y2), box_color, 3)

                        st.session_state.results.append({
                            "original": frame.copy(),
                            "result": result_img.copy(),
                            "text": plate_text,
                            "confidence": confidence,
                            "chars": char_info,
                            "timestamp": timestamp
                        })

            # Hiển thị kết quả lên Streamlit
            camera_placeholder.image(result_img, channels="BGR", use_container_width=True)

            # Độ trễ nhỏ để giảm CPU
            time.sleep(0.05)

        cap.release()
        st.session_state.camera_on = False


# ====== Hiển thị lịch sử ======
if st.session_state.results:
    if st.button("🧹 Xoá lịch sử"):
        st.session_state.results = []
        st.success("Đã xoá lịch sử.")

    st.markdown("🗂️ **Lịch sử nhận diện**")
    df = pd.DataFrame({
        "Biển số": [r["text"] for r in st.session_state.results],
        "Thời điểm": [r["timestamp"] for r in st.session_state.results],
    })
    st.dataframe(df, use_container_width=True)

  
