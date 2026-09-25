# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, chọn năm frame bạn sẽ ưu tiên nếu chỉ có
ngân sách rà năm ảnh. Ghi tên, điểm, thời điểm, thứ tự và lý do; tối thiểu một quyết định phải xét
ảnh gần trùng hoặc trường hợp model không dự đoán được box:

Nếu chỉ có ngân sách rà 5 ảnh, tôi ưu tiên chọn:
1. `frame_0182.jpg` (hạng 1, điểm 0.9591, t=72.8s): Đứng đầu toàn bảng về mức độ bất định $U=0.9182$ và có tỷ lệ box mập mờ cực đại $A=1.0$ (18 box trong dải conf [0.15, 0.50]). Đây là ứng viên giàu thông tin nhất cần người can thiệp.
2. `frame_0369.jpg` (hạng 2, điểm 0.9324, t=147.6s): Mức bất định cao ($U=0.9315$), model dự đoán tới 43 box trong đó 16 box mập mờ ($A=0.8889$), cách cụm thời gian trước hơn 70s.
3. `frame_0380.jpg` (hạng 3, điểm 0.9170, t=152.0s): Bất định cao ($U=0.9340$), cách `frame_0369` 4.4s (> MIN_GAP_S 2s), thuộc đoạn mật độ xe dày đặc.
4. `frame_0326.jpg` (hạng 4, điểm 0.9155, t=130.4s): Bất định cao ($U=0.9310$), 39 box với 15 box mập mờ.
5. `frame_0331.jpg` (hạng 5, điểm 0.9154, t=132.4s): Có tới 47 box và 18 box mập mờ ($A=1.0$). 

Về việc xử lý ảnh gần trùng: `frame_0187.jpg` (hạng 10, điểm 0.8995, t=74.8s) chỉ cách `frame_0182.jpg` đúng 2.0s. Do camera cố định, hai khung hình này có mật độ giao thông và góc chiếu sáng gần như tương đồng. Khi ngân sách bị siết chặt ở 5 ảnh, việc loại bỏ `frame_0187.jpg` để giữ `frame_0182.jpg` giúp bảo toàn công gán nhãn cho các phân đoạn thời gian khác đa dạng hơn.

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV/ảnh contact sheet:

1. `frame_0182.jpg` (hạng 1, điểm 0.9591, U=0.9182, A=1.0, D=1.0): Model phát hiện 28 box nhưng có tới 18 box rơi vào dải bất định (n_ambiguous). Ảnh contact sheet cho thấy nhiều vệt sáng đèn pha ngược chiều gây nhiễu mạnh cho bộ trích xuất đặc trưng.
2. `frame_0099.jpg` (hạng 8, điểm 0.9063, U=0.9460, A=0.7778, D=1.0): Điểm $U$ thuộc nhóm cao nhất pool (0.9460); 29 box dự đoán với 14 box mập mờ. Ảnh cho thấy cụm xe lớn ở làn gần bị cắt mép dưới và xe nhỏ ở xa bị chói sáng.
3. `frame_0107.jpg` (hạng 14, điểm 0.8876, U=0.8752, A=0.8333, D=1.0): 33 box với 15 box mập mờ. Ảnh ghi nhận nhiều vệt phản quang trên mặt đường bị model phân vân giữa nền đường và thân xe.

Một frame có điểm cao nhưng không chọn hoặc một frame có điểm thấp vẫn nên xem, và lý do:

`frame_0372.jpg` xếp hạng 6 với điểm số rất cao 0.9101 ($U=0.9202$, $A=0.8333$), vượt trội so với các ảnh được chọn ở hạng 7, 8, 10, 11, 13, 14, 15. Tuy nhiên, thuật toán chọn mẫu loại bỏ frame này vì nó xuất hiện ở t=148.8s, chỉ cách `frame_0369.jpg` (t=147.6s) vỏn vẹn 1.2s — vi phạm ngưỡng lọc `MIN_GAP_S = 2.0s`. Hai frame cách nhau 1.2s chứa gần như cùng một tập hợp xe chỉ dịch chuyển vài mét. Gán nhãn cả hai sẽ gây trùng lặp gradient, lãng phí nhân lực mà không mang lại tri thức mới cho mô hình.

Điều phép chọn này chưa chứng minh về chất lượng mô hình:

Điểm số cao trong active learning (Uncertainty Sampling) chỉ phản ánh trạng thái lưỡng lự (epistemic/aleatoric confusion) của mô hình hiện tại trên không gian đặc trưng của ảnh đó, chứ **hoàn toàn không bảo đảm rằng việc gán nhãn bổ sung ảnh đó sẽ nâng cao hiệu năng mô hình**. Cụ thể:
1. Độ bất định cao có thể bắt nguồn từ nhiễu cảm biến, chói lóa đèn xe cực đoan hoặc xe bị che khuất gần hết — những trường hợp ngay cả con người cũng khó gán nhãn chuẩn xác, dễ đưa nhãn nhiễu vào dữ liệu train.
2. Chọn mẫu thuần túy theo độ bất định dễ dẫn đến hiện tượng trượt phân bố (sampling bias), tập trung vào các trường hợp biên dị biệt (outliers) mà bỏ qua phân bố tổng thể của tập kiểm thử.
3. Huấn luyện thêm trên các mẫu khó mà không có cơ chế điều hòa hay replay buffer có thể gây hiện tượng quên thảm họa (catastrophic forgetting), làm suy giảm độ chính xác tổng thể thay vì cải thiện mô hình.
