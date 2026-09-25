# Bảng so sánh các vòng

Tập kiểm thử: 20 ảnh, 403 box tham chiếu (bỏ qua 14 box cao dưới 16 px). Ngưỡng IoU 0.5; P, R, F1 tính tại conf 0.25.

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 325 | 0.424 | -0.347 | 0.978 | 0.221 | 0.360 | 0.000 | 0.206 | 0.683 |
