# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Lê Chí Bằng

Công cụ gán nhãn đã dùng: CVAT Docker trên máy cá nhân

Mọi con số truy được từ `reports/rounds_table.md`, `outputs/selection_round1.csv`,
`outputs/metrics_round*.json` hoặc `outputs/round*_diff.md`.

## 1. Dữ liệu và cách chia tập

Tại sao tập chưa gán nhãn (pool) và tập kiểm thử (test set) được chia theo trục thời gian, có vùng
đệm ở giữa, thay vì chia ngẫu nhiên? Nếu chia ngẫu nhiên, số đo trên tập kiểm thử sẽ bị lệch theo
hướng nào, và vì sao?

Trong bài toán thị giác máy tính với camera giám sát cố định (stationary camera), dữ liệu video có tính tự tương quan thời gian cực kỳ mạnh mẽ (high temporal autocorrelation). Một phương tiện giao thông di chuyển qua góc quan sát của camera thường tồn tại liên tục trong nhiều giây. 

Tập chưa gán nhãn (pool) và tập kiểm thử (test set) bắt buộc phải chia theo trục thời gian và có vùng đệm ở giữa (buffer zone) vì:
1. **Loại bỏ rò rỉ dữ liệu theo thời gian (temporal data leakage):** Vùng đệm tạo ra một khoảng trống thời gian đủ lớn để tất cả phương tiện xuất hiện trong tập pool hoàn toàn rời khỏi tầm nhìn của camera trước khi khung hình của tập test bắt đầu ghi nhận.
2. **Đánh giá đúng năng lực khái quát hóa (generalization):** Nếu chia ngẫu nhiên (random split), các khung hình liền kề của cùng một chiếc xe sẽ nằm rải rác ở cả tập huấn luyện và tập kiểm thử. Khi đó, bài toán bị biến tướng từ việc "nhận diện phương tiện mới chưa từng thấy" thành "ghi nhớ hình ảnh chiếc xe đã thấy ở vài khung hình trước" (memorization / object tracking).
3. **Hướng lệch của số đo kiểm thử:** Nếu chia ngẫu nhiên, các chỉ số đánh giá (AP50, Precision, Recall) sẽ bị **lệch lạc quan quá mức (optimistically biased / inflated)**, tạo ra kết quả kiểm thử ảo cao hơn nhiều so với năng lực vận hành thực tế của mô hình khi triển khai.

## 2. Mô hình khởi đầu lạnh (cold start)

Chép dòng vòng 0 từ `rounds_table.md`. Dựa vào `outputs/compare_round0.jpg`, cho biết mô hình khởi
đầu lạnh không khớp nhãn tham chiếu ở những loại xe nào. Độ phủ (recall) theo kích thước xe cho
thấy điều gì? Một trường hợp nào cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai?

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Ở trạng thái khởi đầu lạnh (cold start) sử dụng trọng số tiền huấn luyện COCO, mô hình đạt AP50 là 0.771, Precision@0.25 đạt 0.925 nhưng Recall chỉ đạt 0.489 (bỏ sót hơn 51% tổng lượng xe tham chiếu).

Phân rã độ phủ theo kích thước hình học (recall by size) cho thấy sự chênh lệch rõ rệt:
- **Xe nhỏ (small, diện tích $< 32^2$ px):** Recall chỉ đạt 0.182 (18.2%), tức bỏ sót hơn 81% xe ở xa.
- **Xe vừa (medium, diện tích $32^2 \le \text{area} < 96^2$ px):** Recall đạt 0.547 (54.7%).
- **Xe lớn (large, diện tích $\ge 96^2$ px):** Recall đạt 0.561 (56.1%).

Quan sát ảnh `outputs/compare_round0.jpg` (ví dụ trên `frame_0250.jpg`): Nhãn tham chiếu có 17 xe, mô hình chỉ phát hiện được 6 (TP 6, FP 2, FN 9). Toàn bộ các xe nhỏ đi ngược chiều ở làn xa phía trên cầu vượt bị bỏ sót hoàn toàn. Trong điều kiện ban đêm, đặc trưng hình học thân xe ở xa bị triệt tiêu, chỉ còn lại hai đốm đèn nhỏ — mẫu đặc trưng này không phổ biến trong tập COCO ban ngày, khiến bộ trích xuất đặc trưng của YOLOv8n không đạt ngưỡng kích hoạt.

**Trường hợp cần rà soát lại nhãn tham chiếu:** Nhãn tham chiếu trong bài được sinh tự động bởi mô hình và chưa qua kiểm định thủ công từng khung hình (unverified machine-generated ground truth). Cụ thể, với các xe ở rất xa có chiều cao sát ngưỡng lọc 16 px (`MIN_BOX_H_PX`) hoặc các vệt sáng phản quang trên dải phân cách, nhãn tự động có thể dán nhầm hoặc bỏ sót. Nếu nhãn tham chiếu bỏ sót một xe thực tế trong bóng tối, dự đoán chính xác của mô hình sẽ bị phạt oan thành False Positive (FP). Ngược lại, nếu nhãn tham chiếu đánh dấu vệt đèn đường thành xe, việc mô hình bỏ qua sẽ bị phạt oan thành False Negative (FN). Do đó, cần có chuyên viên kiểm định thủ công (human audit) các trường hợp biên này trước khi kết luận mô hình dự đoán sai.

## 3. Chiến lược chọn mẫu

Giải thích bằng lời công thức `score = W_U·U + W_A·A + W_D·D` và vai trò của `MIN_GAP_S`.
Dẫn ba frame trong `reports/SELECTION.md` và một frame khác để chứng minh cách bạn cân nhắc
độ bất định, ảnh gần trùng và công gán nhãn. Điểm bất định có chứng minh ảnh đó sẽ cải thiện
mô hình không? Vì sao?

Công thức tính điểm ưu tiên chọn mẫu kết hợp giữa mức độ bất định và tính đa dạng thời gian:
$$\text{Score} = W_U \cdot U + W_A \cdot A + W_D \cdot D$$
- **Độ bất định ($U$ - Uncertainty, trọng số $W_U = 0.5$):** Với mỗi box có conf $c \ge 0.05$, độ bất định tính bằng $u(c) = 1 - |2c - 1|$. Hàm này đạt cực đại 1.0 khi $c = 0.5$ (mô hình phân vân nhất giữa hai trạng thái có xe và không có xe), tiệm cận 0 khi $c$ gần 0 hoặc 1. Chỉ số $U$ của frame là trung bình cộng của 5 giá trị $u(c)$ lớn nhất, đại diện cho độ bất định của nhóm box khó nhất trong ảnh.
- **Tỷ lệ box mập mờ ($A$ - Ambiguity, trọng số $W_A = 0.3$):** Là tỷ lệ số box có độ tin cậy nằm trong vùng nghi vấn $0.15 \le c < 0.50$, chuẩn hóa theo số box nghi vấn lớn nhất trong toàn bộ pool ($A = n_{\text{ambiguous}} / \max$).
- **Độ đa dạng thời gian ($D$ - Diversity, trọng số $W_D = 0.2$):** Đo khoảng cách thời gian tới frame đã gán gần nhất, chặn trần ở 10 giây rồi chia cho 10. Ở vòng 1 chưa có frame nào được gán nên $D = 1.0$ cho mọi ảnh.
- **Bonus frame rỗng:** Nếu frame không có dự đoán nào ($empty = True$), $U=A=0$, frame được cộng `EMPTY_BONUS = 0.5` vì bối cảnh đường cao tốc ban đêm luôn có phương tiện; frame rỗng đồng nghĩa mô hình đã bỏ sót toàn bộ (FN tối đa).
- **Vai trò của `MIN_GAP_S = 2.0s`:** Thuật toán áp dụng cơ chế ức chế tham lam (greedy suppression). Hai frame cách nhau dưới 2 giây trên camera tĩnh chứa cùng một nhóm phương tiện và bối cảnh ánh sáng. Việc gán nhãn cả hai sẽ gây lãng phí nhân công và tạo ra các gradient trùng lặp không mang lại tri thức mới.

**Minh chứng từ dữ liệu:**
- Ba frame trong lô 12 ảnh: `frame_0182.jpg` (hạng 1, điểm 0.9591, 18 box ambiguous), `frame_0099.jpg` (hạng 8, điểm 0.9063, 14 box ambiguous) và `frame_0107.jpg` (hạng 14, điểm 0.8876, 15 box ambiguous). Cả ba đều có điểm số $> 0.88$ và tập trung lượng lớn box nằm trong vùng phân vân của mô hình.
- Frame bị loại bỏ: `frame_0372.jpg` xếp hạng 6 với điểm số rất cao 0.9101 ($U=0.9202$, $A=0.8333$), nhưng bị thuật toán loại bỏ vì xuất hiện ở t=148.8s, chỉ cách `frame_0369.jpg` (t=147.6s) đúng 1.2s ($< \text{MIN\_GAP\_S} = 2.0\text{s}$).

**Điểm bất định có chứng minh ảnh đó sẽ cải thiện mô hình không?**
Hoàn toàn không. Điểm bất định cao chỉ phản ánh trạng thái thiếu tự tin của mô hình hiện tại trên mẫu dữ liệu đó. Sự bất định có thể xuất phát từ nhiễu cảm biến, hiện tượng lóa sáng đèn pha cực đoan hoặc xe bị che khuất gần như toàn bộ — những trường hợp mà chính con người cũng khó gán nhãn chính xác. Hơn nữa, huấn luyện tập trung vào các mẫu bất định cực đoan (outliers) mà thiếu cơ chế điều hòa hay replay buffer có thể gây méo mó ranh giới quyết định (decision boundary) và dẫn đến quên thảm họa.

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

**Mức độ can thiệp nhãn (vòng 1):**
Theo `outputs/round1_diff.md`, từ 12 ảnh được chọn, mô hình đề xuất 169 box pre-label. Sau khi rà soát và chỉnh sửa trên CVAT, tổng số box chuẩn hóa là 325:
- **Accepted (giữ nguyên, IoU $\ge 0.85$):** 163 box (tỷ lệ chấp nhận 96.4%).
- **Edited (chỉnh lại kích thước/tọa độ, $0.50 \le \text{IoU} < 0.85$):** 2 box.
- **Deleted (xóa box sai, FP của mô hình):** 4 box.
- **Added (thêm mới box bị bỏ sót, FN của mô hình):** 160 box.
Số box thêm mới (160) xấp xỉ bằng toàn bộ số box mô hình đề xuất ban đầu (169), chứng minh mô hình pre-label bỏ sót gần một nửa số lượng phương tiện thực tế.

**Biến thiên hiệu năng trên tập test:**
- AP50 sụt giảm mạnh từ 0.771 xuống 0.424 ($\Delta = -0.347$).
- Precision tăng lên 0.978 (gần như loại bỏ triệt để FP, chỉ còn 2 FP trên toàn bộ 20 ảnh test).
- Recall sụt giảm nghiêm trọng từ 0.489 xuống 0.221 (tổng FN tăng từ 206 lên 314).
- Phân rã theo kích thước: Recall xe nhỏ tụt từ 0.182 về 0.000 (mất hoàn toàn khả năng phát hiện xe nhỏ); xe vừa tụt từ 0.547 xuống 0.206; duy nhất xe lớn cải thiện từ 0.561 lên 0.683.
- **Nguyên nhân kỹ thuật:** Đây là minh chứng rõ rệt của hiện tượng **quên thảm họa (catastrophic forgetting)** kết hợp **overfitting cục bộ**. Việc huấn luyện 50 epoch trên vỏn vẹn 12 ảnh ban đêm (325 box) mà không cố định backbone hay sử dụng bộ đệm dữ liệu (replay buffer) từ COCO khiến mô hình bị co cụm ranh giới phân loại. Mô hình học thuộc đặc trưng của các xe lớn, rõ nét ở làn gần và đẩy ngưỡng kích hoạt lên cao, dẫn đến thái độ dự đoán cực kỳ thận trọng (high precision, low recall), từ chối nhận diện các tín hiệu yếu của xe nhỏ ở xa.

**Minh chứng trên ảnh so sánh:**
Quan sát `outputs/compare_round1.jpg` trên `frame_0050.jpg`: Ở vòng 0, mô hình bắt được 11 xe (TP 11, FP 2, FN 7). Sau khi fine-tune ở vòng 1, số xe phát hiện được giảm xuống chỉ còn 6 xe (TP 6, FP 0, FN 12). Toàn bộ các xe nhỏ ở làn xa phía trên cầu vượt đã biến mất khỏi dự đoán của mô hình.

**Phân biệt ba góc nhìn:**
1. *Quan sát độc lập (`BLIND_SCAN.md`):* Đếm thuần túy bằng mắt thường trên `frame_0099.jpg` được 25 xe, nhận diện điểm mù ở vùng mép cắt và vùng bị che khuất mà không chịu định kiến từ gợi ý của AI.
2. *Lỗi pre-label đã sửa (`REVIEW_LOG.csv` và `round1_diff.md`):* Đối chiếu thực tế và can thiệp nhãn theo guideline — kéo rộng box ở `frame_0326.jpg` do AI chỉ khoanh đốm đèn; xóa box ở `frame_0227.jpg` do AI nhận nhầm vệt phản quang trên đường; thêm box ở `frame_0369.jpg` do AI bỏ sót xe ngược chiều.
3. *Kết quả mô hình sau train:* Cho thấy việc bổ sung 160 box vào 12 ảnh huấn luyện chưa đủ độ phong phú về không gian đặc trưng để mô hình khái quát hóa, ngược lại làm mất đi tri thức tiền huấn luyện.

**Mô tả ca khó theo `GUIDELINE_LABEL.md`:**
Trường hợp xe bị cắt mép ảnh (truncated, ví dụ xe ở góc dưới bên phải `frame_0099.jpg`): Guideline quy định chỉ vẽ khung ôm sát phần thân xe thực sự nhìn thấy được trong khung hình (đuôi xe và cụm đèn hậu), tuyệt đối không phỏng đoán kích thước xe để vẽ box vượt ra ngoài giới hạn ảnh $[0, 1]$. Đồng thời, với xe đi ngược chiều ban đêm, khung phải bao trọn phần thân xe ước lượng từ vệt sáng phản chiếu, không được khoanh vệt đèn pha rọi dài trên mặt đường.

## 5. Kết luận và giới hạn

Kết quả vòng này so với cold start ra sao? Vì sao bạn dừng hoặc tiếp tục? Đề xuất hai ca còn yếu
hoặc bất định cho vòng sau, kèm chi phí rà nhãn và nguy cơ ảnh gần trùng. Tập kiểm thử chỉ 20 ảnh,
có luật bỏ qua xe quá nhỏ và nhãn tham chiếu do mô hình tạo chưa được rà thủ công; các giới hạn đó
ảnh hưởng thế nào đến kết luận? Nếu AP50 giảm, bạn sẽ kiểm tra điều gì trước khi train thêm?

**Đánh giá tổng thể và quyết định:**
Sau vòng 1, AP50 giảm từ 0.771 xuống 0.424. Mặc dù điểm số sụt giảm, tôi quyết định **tiếp tục vòng 2**. Sự sụt giảm này là hệ quả tất yếu khi fine-tune số lượng nhỏ mẫu dữ liệu (12 ảnh) trên một mô hình đã học trên hàng triệu ảnh COCO mà chưa có chiến lược điều hòa tham số phù hợp. Dừng lại ở vòng 1 đồng nghĩa chấp nhận một mô hình bị suy thoái do thiếu hụt dữ liệu.

**Đề xuất hai ca khó cho vòng sau:**
1. *Xe nhỏ ở cự ly xa (chỉ hiển thị 2 đốm đèn):* Recall hiện tại là 0.000. Chi phí gán nhãn cho nhóm này rất cao vì nhân viên phải phóng to điểm ảnh để phân biệt đốm đèn xe với đèn chiếu sáng đường. Nguy cơ nhầm lẫn dẫn đến việc nạp nhãn nhiễu vào mô hình.
2. *Xe bị che khuất một phần (occluded) trong các cụm giao thông dày đặc:* Đòi hỏi công gán nhãn tỉ mỉ để tách biệt ranh giới giữa hai xe đi sát nhau. Nguy cơ ảnh gần trùng ở trường hợp này rất lớn do các dòng xe ùn ứ thường di chuyển chậm; cần duy trì nghiêm ngặt `MIN_GAP_S` để tránh gán nhãn các cảnh lặp lại.

**Giới hạn thực nghiệm:**
- **Quy mô tập test nhỏ (20 ảnh, 403 box):** Cỡ mẫu quá nhỏ dẫn đến phương sai thống kê cao. Việc biến thiên điểm số AP50 có thể bị chi phối mạnh bởi một vài khung hình cá biệt thay vì phản ánh xu hướng thực sự của mô hình.
- **Quy tắc bỏ qua xe quá nhỏ (< 16 px) và nhãn tham chiếu chưa qua kiểm định thủ công:** Nhãn tham chiếu do máy tự tạo có thể chứa lỗi hệ thống (bỏ sót xe tối màu hoặc nhận nhầm vệt sáng). Việc đánh giá mô hình trên một tập "ground truth" chưa hoàn hảo làm giảm độ tin cậy tuyệt đối của chỉ số AP50.

**Quy trình kiểm tra (QC) khi AP50 giảm trước khi huấn luyện tiếp:**
1. *Kiểm tra tính toàn vẹn của nhãn (Label sanity check):* Xác minh toàn bộ file `.txt` trong `labels/round1/` có đúng mã lớp 0, tọa độ chuẩn hóa $[0, 1]$, không có box rỗng hay box bị đảo chiều tọa độ.
2. *Kiểm soát siêu tham số huấn luyện (Hyperparameter tuning):* Giảm số epoch (từ 50 xuống 15–20) để hạn chế overfitting; giảm learning rate hoặc đóng băng các tầng trích xuất đặc trưng (freeze backbone) để bảo tồn tri thức COCO; bổ sung kỹ thuật replay (trộn thêm ảnh gốc) nhằm chống quên thảm họa.
3. *Kiểm tra phân bố lô chọn ở vòng 2:* Đảm bảo khoảng cách thời gian và độ đa dạng không gian được duy trì, loại bỏ các frame bị chói lóa bất thường trước khi đưa vào tập huấn luyện tiếp theo.
