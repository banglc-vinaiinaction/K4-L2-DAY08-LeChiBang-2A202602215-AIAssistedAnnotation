# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, chọn năm frame bạn sẽ ưu tiên nếu chỉ có
ngân sách rà năm ảnh. Ghi tên, điểm, thời điểm, thứ tự và lý do; tối thiểu một quyết định phải xét
ảnh gần trùng hoặc trường hợp model không dự đoán được box:

Nếu chỉ được sửa 5 ảnh, em chọn frame_0182.jpg (hạng 1, điểm 0.959), frame_0369.jpg (hạng 2, điểm 0.932), frame_0380.jpg (hạng 3, điểm 0.917), frame_0326.jpg (hạng 4, điểm 0.916) và frame_0331.jpg (hạng 5, điểm 0.915). Năm ảnh này đứng đầu danh sách. frame_0182.jpg (t=72.8s) và frame_0187.jpg (t=74.8s) cách nhau khoảng 2 giây, cảnh gần giống nhau, nên em không lấy cả hai mà chỉ giữ frame_0182.jpg vì điểm cao hơn.

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV/ảnh contact sheet:

Trong 12 ảnh AI đã chọn, em nhìn frame_0182.jpg (hạng 1, điểm 0.959, 28 box, 18 ambiguous), frame_0099.jpg (hạng 8, điểm 0.906, 29 box, 14 ambiguous) và frame_0107.jpg (hạng 14, điểm 0.888, 33 box, 15 ambiguous). Cả ba đều có điểm trên 0.88 và AI khoanh nhiều xe nhưng còn nhiều khung chưa chắc (n_ambiguous cao).

Một frame có điểm cao nhưng không chọn hoặc một frame có điểm thấp vẫn nên xem, và lý do:

frame_0372.jpg hạng 6, điểm 0.910, cao hơn vài ảnh đã được chọn, nhưng AI bỏ qua vì nó cách frame_0369.jpg chỉ 1.2 giây (t=148.8s so với t=147.6s). Hai ảnh gần như cùng một cảnh, sửa cả hai thì tốn công mà ít học thêm được gì.

Điều phép chọn này chưa chứng minh về chất lượng mô hình:

Điểm cao chỉ nghĩa là AI đang phân vân. Nó chưa chứng minh sửa ảnh đó xong thì AI sẽ nhận xe tốt hơn.
