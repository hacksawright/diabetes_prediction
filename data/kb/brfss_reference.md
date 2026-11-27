# Mô hình BRFSS trong Diabet

## Ý nghĩa điểm phần trăm

Phần trăm trên báo cáo Diabet là điểm đầu ra của mạng neuron BRFSS cho thông tin vừa nhập. Đây là điểm tham khảo. Nó không phải kết quả xét nghiệm HbA1c, xác suất mắc bệnh đã hiệu chuẩn hay kết quả chẩn đoán.

## Dữ liệu và phạm vi

Mô hình `brfss_web5_mlp_v1` học từ tập chỉ báo sức khỏe BRFSS 2015 đã xử lý. Nhãn dựa trên tình trạng đái tháo đường được báo cáo trong khảo sát người trưởng thành tại Hoa Kỳ. Khảo sát đo đặc điểm và tình trạng tại thời điểm thu thập, nên bản demo không dự đoán diễn tiến đường huyết hay thời điểm mắc bệnh trong tương lai.

## Đầu vào

Năm đầu vào gồm nhóm tuổi BRFSS, BMI, sức khỏe tự đánh giá, từng được nhân viên y tế báo huyết áp cao và từng được báo cholesterol cao. Glucose, HbA1c và kết quả đo từ cảm biến không nằm trong đầu vào của phiên bản này.

## Hệ thống tạo báo cáo thế nào?

Mạng neuron tạo điểm tham khảo. Rule-Based Agent ghi lưu ý kỹ thuật nếu có, chẳng hạn BMI ngoài khoảng quan sát khi huấn luyện. KG/RAG tìm tài liệu để giải thích khái niệm sức khỏe. Báo cáo kết hợp các phần này, còn thông tin chuyên môn có thể được tra cứu bằng khung chat.
