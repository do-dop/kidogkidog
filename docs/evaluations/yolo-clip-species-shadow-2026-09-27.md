# YOLO + CLIP species resolver shadow dry-run

Eligible 240개 frame만 읽은 read-only 실행이다. Chroma, MySQL, scenes, behavior_events는 읽거나 쓰지 않았고 ineligible frame은 처리하지 않았다.

- 처리: 240/240, 실패: 0
- Resolver 적용 detection: 311
- YOLO→CLIP 종 변경: 37 (dog→cat 12, cat→dog 25)
- 0.4–0.5 confidence detection 회복: 36
- 이전 dry-run 대비 raw YOLO 불일치 frame: 0
- 평가 Policy A(10% padding, prompt A) 대비 CLIP label 불일치 frame: 0

상세 JSON에는 raw detection, detection별 species_resolution, 최종 object_labels, frame별 평가 비교가 들어 있다.
