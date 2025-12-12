Dự án thị giác máy tính hỗ trợ phát hiện và nhận dạng biển số xe Việt Nam sử dụng mô hình YOLOv8
kết hợp EasyOCR để trích xuất ký tự. Hệ thống được tối ưu cho nhiều điều kiện ánh sáng và các loại biển số khác nhau.
Tính năng chính
  Phát hiện biển số xe bằng mô hình YOLOv8/YOLOv11 với độ chính xác cao
  Trích xuất ký tự từ biển số bằng EasyOCR
  Thu thập & gán nhãn bộ dữ liệu riêng
  Xử lý tốt ảnh ngày/đêm và nhiều góc chụp
  Pipeline nhận diện end-to-end
1. Phát hiện biển số (Detection)

  Huấn luyện với YOLOv8 và YOLOv11
  
  Kích thước ảnh: 640
  
  Tối ưu batch size theo GPU
  
  Augmentation dữ liệu để tăng khả năng tổng quát

  Đánh giá bằng: mAP50, mAP50-95

2. Nhận dạng ký tự (OCR)

  Trích xuất ký tự bằng EasyOCR
  
  Tiền và hậu xử lý để tăng độ chính xác
  
  Ghép chuỗi ký tự theo đúng định dạng biển số Việt Nam
