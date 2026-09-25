# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Lê Chí Bằng

Công cụ gán nhãn đã dùng: CVAT

Mọi con số truy được từ `reports/rounds_table.md`, `outputs/selection_round1.csv`,
`outputs/metrics_round*.json` hoặc `outputs/round*_diff.md`.

## 1. Dữ liệu và cách chia tập

Tại sao tập chưa gán nhãn (pool) và tập kiểm thử (test set) được chia theo trục thời gian, có vùng
đệm ở giữa, thay vì chia ngẫu nhiên? Nếu chia ngẫu nhiên, số đo trên tập kiểm thử sẽ bị lệch theo
hướng nào, và vì sao?

Camera đứng một chỗ, một chiếc xe nằm trong hình vài giây liên tục. Ảnh học và ảnh kiểm tra phải cách nhau theo thời gian để không trùng cảnh. Nếu trộn ngẫu nhiên, cùng một xe có thể vừa được AI học vừa được dùng để chấm. Khi đó điểm sẽ đẹp hơn sự thật vì AI đã thấy xe đó rồi, không phải nhận xe mới.

## 2. Mô hình khởi đầu lạnh (cold start)

Chép dòng vòng 0 từ `rounds_table.md`. Dựa vào `outputs/compare_round0.jpg`, cho biết mô hình khởi
đầu lạnh không khớp nhãn tham chiếu ở những loại xe nào. Độ phủ (recall) theo kích thước xe cho
thấy điều gì? Một trường hợp nào cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai?

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Điểm khớp khung AP50 là 0.771. Xe nhỏ (ở xa) chỉ được tìm thấy 18.2%, xe vừa 54.7%, xe lớn (ở gần) 56.1%. Nghĩa là xe ở xa camera bị bỏ sót nhiều hơn xe ở gần. Trong ảnh compare_round0.jpg, frame_0250 có 17 box tham chiếu nhưng cold start chỉ tìm được 6 (TP 6, FP 2, FN 9); nhiều xe nhỏ ở làn xa phía trên ảnh bị bỏ qua hoàn toàn. Tuy nhiên, nhãn tham chiếu dùng để chấm cũng do máy vẽ, chưa có người xem từng khung, nên có thể nhãn chấm sai chứ không phải AI sai.

## 3. Chiến lược chọn mẫu

Giải thích bằng lời công thức `score = W_U·U + W_A·A + W_D·D` và vai trò của `MIN_GAP_S`.
Dẫn ba frame trong `reports/SELECTION.md` và một frame khác để chứng minh cách bạn cân nhắc
độ bất định, ảnh gần trùng và công gán nhãn. Điểm bất định có chứng minh ảnh đó sẽ cải thiện
mô hình không? Vì sao?

Mỗi ảnh có một điểm tổng hợp. Khoảng một nửa điểm (W_U) đến từ độ bất định — AI không chắc ảnh đó có xe hay không. Ba phần mười (W_A) đến từ tỉ lệ khung mà AI còn lưỡng lự, tức AI vẽ nhiều khung nhưng chưa tự tin. Hai phần mười (W_D) đánh giá ảnh đó có khác thời gian với các ảnh đã chọn hay không. Hai ảnh trong cùng một lô phải cách nhau ít nhất 2 giây (MIN_GAP_S), vì camera đứng yên nên ảnh sát nhau gần như giống hệt, sửa cả hai thì tốn công mà ít học thêm được gì.

Trong SELECTION.md, em nhìn ba ảnh AI đã chọn: frame_0182.jpg (hạng 1, điểm 0.959, 18 ambiguous), frame_0099.jpg (hạng 8, điểm 0.906, 14 ambiguous) và frame_0107.jpg (hạng 14, điểm 0.888, 15 ambiguous). Cả ba đều có điểm trên 0.88 và AI khoanh nhiều xe nhưng còn nhiều khung chưa chắc. frame_0372.jpg đứng hạng 6 với điểm 0.910, cao hơn vài ảnh đã được chọn, nhưng AI bỏ qua vì nó cách frame_0369.jpg chỉ 1.2 giây — hai ảnh gần như cùng một cảnh. Điểm cao không có nghĩa sửa ảnh đó sẽ làm AI giỏi hơn, nó chỉ nghĩa là AI đang phân vân.

## 4. Các vòng học chủ động (active learning)

Chép bảng từ `rounds_table.md`. Với mỗi vòng, trình bày:

- mức độ bạn đã sửa nhãn gợi ý (số box giữ nguyên, chỉnh sửa, xoá, thêm mới, lấy từ
  `outputs/round*_diff.md`);
- AP50 thay đổi bao nhiêu so với khởi đầu lạnh và so với vòng trước;
- nhóm xe nào tốt lên hoặc xấu đi theo số đo trên cùng tập test.

Dựa vào các ảnh `compare_round*.jpg`, chỉ ra một ca kết quả đổi sau fine-tune (tốt hơn hoặc xấu
đi), cùng lý do có thể kiểm. Dùng `BLIND_SCAN.md`, `REVIEW_LOG.csv` và `round1_diff.md` phân biệt
quan sát độc lập, lỗi pre-label đã sửa và kết quả mô hình sau train. Mô tả một ca khó theo guideline.

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 325 | 0.424 | -0.347 | 0.978 | 0.221 | 0.360 | 0.000 | 0.206 | 0.683 |

Vòng 1 dùng 12 ảnh, sau khi sửa có 325 box. Theo round1_diff.md: em giữ nguyên 163 khung, kéo lại 2, xóa 4, thêm mới 160. Model đề xuất 169 box nhưng bỏ sót rất nhiều xe nhỏ nên em phải thêm gần gấp đôi.

AP50 giảm từ 0.771 xuống 0.424 (giảm 0.347). Recall giảm từ 0.489 xuống 0.221. Xe nhỏ recall rớt về 0.000, xe vừa từ 0.547 xuống 0.206. Chỉ có xe lớn tăng từ 0.561 lên 0.683. Model sau fine-tune trở nên rất thận trọng — precision tăng lên 0.978 nhưng bỏ sót hầu hết xe.

Trong compare_round1.jpg, frame_0050 cho thấy sự thay đổi rõ: cold start tìm được 11 xe (TP 11, FP 2, FN 7), nhưng round 1 chỉ tìm được 6 xe (TP 6, FP 0, FN 12). Model mất khả năng nhận xe nhỏ và xe vừa ở xa, chỉ giữ lại xe lớn ở gần.

Ba việc cần phân biệt: trong BLIND_SCAN.md, em nhìn frame_0099.jpg bằng mắt và đếm 25 xe, ghi nhận xe bị cắt mép và xe bị che là hai vị trí AI dễ sai. Trong REVIEW_LOG.csv, em ghi lại ba sửa cụ thể — kéo rộng box xe rất xa ở frame_0326.jpg vì AI chỉ khoanh đèn, xóa vệt sáng đèn trên mặt đường ở frame_0227.jpg vì không phải xe, và thêm xe nhỏ làn ngược ở frame_0369.jpg mà AI bỏ sót. Sau khi AI học lại, kết quả cho thấy 12 ảnh với 325 box chưa đủ để AI giữ được kiến thức từ COCO, dẫn đến catastrophic forgetting.

## 5. Kết luận và giới hạn

Kết quả vòng này so với cold start ra sao? Vì sao bạn dừng hoặc tiếp tục? Đề xuất hai ca còn yếu
hoặc bất định cho vòng sau, kèm chi phí rà nhãn và nguy cơ ảnh gần trùng. Tập kiểm thử chỉ 20 ảnh,
có luật bỏ qua xe quá nhỏ và nhãn tham chiếu do mô hình tạo chưa được rà thủ công; các giới hạn đó
ảnh hưởng thế nào đến kết luận? Nếu AP50 giảm, bạn sẽ kiểm tra điều gì trước khi train thêm?

AP50 vòng 1 là 0.424, giảm 0.347 so với cold start 0.771. Điểm giảm mạnh. Em chọn tiếp tục thêm một vòng nữa vì 12 ảnh chưa đủ để AI học cảnh ban đêm mà không quên kiến thức cũ, và round 2 đã có sẵn 12 ảnh mới được chọn.

Hai chỗ còn yếu: thứ nhất, xe ở rất xa chỉ còn hai chấm đèn, AI hiện không nhận được loại này (recall small = 0.000); thứ hai, xe bị cắt mép ảnh — AI thường bỏ sót hoặc vẽ khung không ôm hết thân xe. Sửa thêm ảnh cho hai trường hợp này sẽ tốn thời gian vì phải kéo khung cẩn thận, và không nên chọn hai ảnh sát nhau vì chúng gần như cùng một cảnh.

Giới hạn: tập kiểm thử chỉ có 20 ảnh, xe quá nhỏ (cao dưới 16 px) không được tính, và nhãn tham chiếu chưa được người kiểm lại. Những giới hạn đó khiến kết luận chưa chắc chắn — điểm có thể dao động lớn chỉ vì một vài ảnh test. Vì AP50 giảm, trước khi cho AI học thêm, em cần xem lại khung đã sửa ở vòng 1: kiểm tra có khung nào vẽ sai quy cách không, có nhãn nào bị lệch class không, và đảm bảo file data.yaml đúng cấu trúc.
