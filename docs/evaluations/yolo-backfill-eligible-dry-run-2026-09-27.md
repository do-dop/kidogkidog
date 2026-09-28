# YOLOv8s eligible-frame backfill dry-run

- Input: `docs/evaluations/yolo-backfill-eligible-2026-09-27.json`
- Model: `yolov8s.pt` (SHA-256 `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a`), Ultralytics `8.4.33`, device `mps`
- Thresholds: detections ≥ 0.4; object_labels ≥ 0.5
- Processed: 240/240; failures: 0

## 전체 변경 통계

| Metric | Count |
|---|---:|
| total | 240 |
| unchanged | 57 |
| changed | 183 |
| empty → detected | 31 |
| detected → empty | 2 |
| labels added (class occurrences) | 234 |
| labels removed (class occurrences) | 62 |
| old cat → new dog | 9 |
| old dog → new cat | 11 |
| person removed | 2 |
| person added | 4 |
| empty → cat | 7 |
| empty → dog | 15 |
| empty → person | 1 |
| empty → animal | 22 |
| animal → empty | 1 |
| dog → person | 0 |
| cat → person | 0 |
| changed ratio | 76.2% |

### Labels added/removed by class

- Added: `{'bed': 6, 'bottle': 3, 'bowl': 71, 'cat': 21, 'chair': 31, 'couch': 20, 'cow': 4, 'dining table': 13, 'dog': 57, 'horse': 3, 'person': 4, 'tv': 1}`
- Removed: `{'bed': 10, 'cat': 10, 'chair': 7, 'couch': 12, 'cow': 1, 'dog': 11, 'horse': 4, 'person': 2, 'tv': 5}`

## Video별 통계

| video_id | total | unchanged | changed | changed ratio | empty→detected | detected→empty | cat→dog | dog→cat | person− | person+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IMG_8450_2 | 33 | 6 | 27 | 81.8% | 1 | 0 | 1 | 6 | 0 | 0 |
| dog_drinking_water | 89 | 22 | 67 | 75.3% | 18 | 0 | 5 | 5 | 2 | 1 |
| dog_escape | 28 | 14 | 14 | 50.0% | 4 | 2 | 0 | 0 | 0 | 1 |
| two_dogs | 90 | 15 | 75 | 83.3% | 8 | 0 | 3 | 0 | 0 | 2 |

## cat / dog / person 존재 여부 변화

| class | old=1,new=1 | old=1,new=0 | old=0,new=1 | old=0,new=0 |
|---|---:|---:|---:|---:|
| cat | 34 | 10 | 21 | 175 |
| dog | 85 | 11 | 57 | 87 |
| person | 12 | 2 | 4 | 222 |

## 이전 107개 표본 label 일치 확인

- Matched frame IDs: 107/107
- Identical label sets: 34/107
- Mismatches: 73

## 기존 시각 확인 GT 표본에서 old label 변화

| class | reviewed | old correct | new correct | recovered misses | lost true labels | removed false positives | new false positives |
|---|---:|---:|---:|---:|---:|---:|---:|
| cat | 107 | 94 | 96 | 4 | 1 | 5 | 6 |
| dog | 106 | 56 | 69 | 15 | 1 | 6 | 7 |
| person | 106 | 103 | 103 | 0 | 0 | 2 | 2 |

GT subset actual animal frames: 94; old animal label absent: 48; new recovered: 19; still missed: 29; new person label without detected animal: 3

## 주요 label 변경 사례

- `two_dogs__two_dogs_000_frame_000005` (two_dogs @ 8.0s): [] → ['dog'] [label_added, empty_to_detected, empty_to_dog, empty_to_animal]. Detections: dog 0.533 bbox=[182.466064453125, 269.7972106933594, 1164.355712890625, 943.4221801757812]; couch 0.494 bbox=[0.0, 1.438751220703125, 1674.06005859375, 1079.2435302734375]
- `two_dogs__two_dogs_000_frame_000007` (two_dogs @ 12.0s): [] → ['couch', 'dog'] [label_added, empty_to_detected, empty_to_dog, empty_to_animal]. Detections: dog 0.769 bbox=[859.8682250976562, 170.558349609375, 1564.0584716796875, 672.875244140625]; couch 0.727 bbox=[4.59393310546875, 2.565948486328125, 1751.3109130859375, 1052.88720703125]; bed 0.410 bbox=[2.473480224609375, 576.9546508789062, 1015.0301513671875, 1067.6041259765625]
- `two_dogs__two_dogs_000_frame_000037` (two_dogs @ 72.0s): ['cat', 'dog', 'tv'] → ['chair', 'dog', 'tv'] [label_added, label_removed, cat_to_dog]. Detections: dog 0.932 bbox=[674.0286865234375, 515.3919067382812, 984.652099609375, 1074.3817138671875]; dog 0.886 bbox=[1253.4254150390625, 753.0732421875, 1728.3782958984375, 1075.499267578125]; tv 0.848 bbox=[373.2392578125, 228.6137237548828, 603.0348510742188, 394.6668395996094]; chair 0.572 bbox=[1081.8182373046875, 447.5116882324219, 1334.1650390625, 867.9585571289062]; chair 0.492 bbox=[0.88916015625, 436.5444030761719, 250.462890625, 646.2921752929688]; chair 0.462 bbox=[1734.8433837890625, 335.5295104980469, 1919.011962890625, 1071.3779296875]; couch 0.461 bbox=[1736.37451171875, 335.37652587890625, 1919.0009765625, 1068.5946044921875]
- `two_dogs__two_dogs_000_frame_000040` (two_dogs @ 78.0s): ['tv'] → ['couch', 'person', 'tv'] [label_added, person_added]. Detections: tv 0.744 bbox=[377.1103820800781, 230.20010375976562, 606.7142944335938, 405.03936767578125]; person 0.575 bbox=[738.2781372070312, 7.859893798828125, 1914.6749267578125, 1061.7633056640625]; couch 0.556 bbox=[0.1761474609375, 438.0355224609375, 241.21153259277344, 667.966552734375]
- `two_dogs__two_dogs_000_frame_000046` (two_dogs @ 90.0s): ['cat', 'tv'] → ['chair', 'dog', 'tv'] [label_added, label_removed, cat_to_dog]. Detections: dog 0.865 bbox=[0.7408447265625, 528.1151733398438, 426.46966552734375, 1067.8187255859375]; tv 0.850 bbox=[372.8343811035156, 227.43228149414062, 601.6832885742188, 396.5450134277344]; chair 0.688 bbox=[1735.543212890625, 337.7982482910156, 1918.951904296875, 1068.08056640625]
- `two_dogs__two_dogs_000_frame_000049` (two_dogs @ 96.0s): [] → ['dog'] [label_added, empty_to_detected, empty_to_dog, empty_to_animal]. Detections: dog 0.710 bbox=[676.9390869140625, 179.72991943359375, 1622.926025390625, 1042.1048583984375]
- `two_dogs__two_dogs_000_frame_000054` (two_dogs @ 106.0s): ['cat', 'dog', 'tv'] → ['chair', 'dog', 'tv'] [label_added, label_removed, cat_to_dog]. Detections: dog 0.840 bbox=[0.34698486328125, 563.15185546875, 1103.4942626953125, 1073.660888671875]; tv 0.761 bbox=[369.2607727050781, 228.48655700683594, 602.84521484375, 410.3094177246094]; chair 0.585 bbox=[1735.6494140625, 337.34527587890625, 1918.98486328125, 1069.4256591796875]; chair 0.433 bbox=[1078.0572509765625, 444.6251220703125, 1336.281005859375, 865.119384765625]; dog 0.420 bbox=[1045.118896484375, 839.674072265625, 1627.0599365234375, 1074.7144775390625]; couch 0.407 bbox=[0.7039947509765625, 439.5806579589844, 238.51730346679688, 691.7679443359375]
- `two_dogs__two_dogs_000_frame_000078` (two_dogs @ 154.0s): ['chair', 'couch', 'tv'] → ['chair', 'couch', 'dining table', 'dog', 'person', 'tv'] [label_added, person_added]. Detections: chair 0.836 bbox=[1421.3094482421875, 407.3346862792969, 1574.3397216796875, 701.7490234375]; chair 0.816 bbox=[1634.945068359375, 451.80908203125, 1883.185546875, 843.841552734375]; chair 0.804 bbox=[1507.4754638671875, 431.401611328125, 1710.5628662109375, 776.7584838867188]; person 0.749 bbox=[1046.2503662109375, 179.74717712402344, 1157.386962890625, 446.4041748046875]; dog 0.626 bbox=[1151.4390869140625, 512.3351440429688, 1313.44677734375, 662.0908813476562]; tv 0.611 bbox=[294.79827880859375, 79.31460571289062, 626.8632202148438, 305.8779602050781]; couch 0.581 bbox=[0.870849609375, 579.3389892578125, 544.4656372070312, 1067.833251953125]; dining table 0.521 bbox=[1453.4989013671875, 373.9040222167969, 1919.05078125, 827.3228759765625]; chair 0.487 bbox=[0.765289306640625, 575.49951171875, 541.1380615234375, 1069.2276611328125]; bed 0.469 bbox=[280.9679260253906, 565.3532104492188, 643.2971801757812, 670.2118530273438]; dog 0.434 bbox=[1088.443359375, 434.64752197265625, 1375.294921875, 625.7807006835938]
- `IMG_8450_2__petcam_20260926_173123_000_frame_6.50` (IMG_8450_2 @ 6.5s): [] → ['bed', 'cat'] [label_added, empty_to_detected, empty_to_cat, empty_to_animal]. Detections: cat 0.869 bbox=[577.0997924804688, 568.5198974609375, 1015.771240234375, 985.7499389648438]; cat 0.833 bbox=[620.7088623046875, 480.280517578125, 1009.2943725585938, 739.9962158203125]; bed 0.531 bbox=[909.35888671875, 291.2328186035156, 1918.9588623046875, 1073.427001953125]
- `IMG_8450_2__petcam_20260926_173123_000_frame_8.50` (IMG_8450_2 @ 8.5s): ['dog'] → ['cat'] [label_added, label_removed, dog_to_cat]. Detections: cat 0.887 bbox=[547.4594116210938, 494.5295104980469, 897.9644165039062, 925.3267822265625]; cat 0.786 bbox=[675.284423828125, 502.893310546875, 1068.1925048828125, 755.3676147460938]
- `IMG_8450_2__petcam_20260926_173123_000_frame_12.00` (IMG_8450_2 @ 12.0s): ['dog'] → ['bowl', 'cat'] [label_added, label_removed, dog_to_cat]. Detections: cat 0.942 bbox=[721.8348999023438, 490.26654052734375, 1082.378662109375, 740.9039306640625]; cat 0.896 bbox=[621.8638916015625, 522.8526000976562, 955.7008666992188, 918.5953369140625]; bowl 0.618 bbox=[292.8049621582031, 638.68505859375, 467.6076965332031, 785.4948120117188]
- `IMG_8450_2__petcam_20260926_173123_000_frame_14.00` (IMG_8450_2 @ 14.0s): ['dog'] → ['bowl', 'cat'] [label_added, label_removed, dog_to_cat]. Detections: cat 0.946 bbox=[502.5103454589844, 516.5958251953125, 975.2345581054688, 899.35107421875]; cat 0.862 bbox=[718.8256225585938, 460.3310241699219, 1072.90380859375, 725.3113403320312]; bowl 0.859 bbox=[287.110107421875, 617.965087890625, 461.66619873046875, 766.9610595703125]
- `IMG_8450_2__petcam_20260926_173123_000_frame_16.00` (IMG_8450_2 @ 16.0s): ['cat', 'dog'] → ['bowl', 'cat'] [label_added, label_removed, dog_to_cat]. Detections: cat 0.895 bbox=[542.9650268554688, 647.5499877929688, 820.9304809570312, 1076.62890625]; cat 0.871 bbox=[633.0391845703125, 486.9396057128906, 1081.47119140625, 714.912841796875]; bowl 0.627 bbox=[286.1243591308594, 607.9392700195312, 462.064453125, 756.90673828125]
- `IMG_8450_2__petcam_20260926_173123_000_frame_26.00` (IMG_8450_2 @ 26.0s): ['cat', 'dog'] → ['bowl', 'cat'] [label_added, label_removed, dog_to_cat]. Detections: cat 0.909 bbox=[655.1220703125, 372.0868835449219, 1060.83154296875, 579.68212890625]; cat 0.812 bbox=[620.1570434570312, 651.7262573242188, 914.0166015625, 1067.412841796875]; bowl 0.721 bbox=[321.056640625, 482.36334228515625, 493.0809631347656, 628.64208984375]; bed 0.451 bbox=[1444.7733154296875, 145.14431762695312, 1918.880615234375, 934.5422973632812]
- `IMG_8450_2__petcam_20260926_173123_000_frame_46.00` (IMG_8450_2 @ 46.0s): ['cat', 'dog'] → ['bowl', 'cat'] [label_added, label_removed, dog_to_cat]. Detections: cat 0.888 bbox=[820.106689453125, 383.6279296875, 1254.6058349609375, 589.02197265625]; cat 0.820 bbox=[751.3084716796875, 749.3704833984375, 1140.1856689453125, 1079.08447265625]; bowl 0.592 bbox=[480.42864990234375, 504.7731628417969, 649.6246948242188, 649.46435546875]
- `IMG_8450_2__petcam_20260926_173123_000_frame_66.00` (IMG_8450_2 @ 66.0s): ['cat'] → ['bowl', 'dog'] [label_added, label_removed, cat_to_dog]. Detections: dog 0.882 bbox=[850.2628784179688, 650.2279052734375, 1162.366455078125, 1004.0178833007812]; bowl 0.755 bbox=[579.7307739257812, 385.9215393066406, 742.5122680664062, 527.221923828125]; dog 0.529 bbox=[726.741943359375, 290.9590759277344, 1007.9437866210938, 661.01220703125]; cat 0.470 bbox=[727.321044921875, 291.4517822265625, 1009.4716186523438, 663.325439453125]; chair 0.423 bbox=[247.71775817871094, 2.2322845458984375, 509.7654724121094, 675.6492919921875]
- `dog_escape__petcam_20260926_173415_000_frame_30.00` (dog_escape @ 30.0s): [] → ['cat'] [label_added, empty_to_detected, empty_to_cat, empty_to_animal]. Detections: cat 0.775 bbox=[359.44573974609375, 45.13398742675781, 952.907958984375, 514.1370849609375]
- `dog_escape__petcam_20260926_173415_000_frame_48.00` (dog_escape @ 48.0s): [] → ['person'] [label_added, empty_to_detected, person_added, empty_to_person]. Detections: person 0.648 bbox=[600.3314819335938, 736.2921752929688, 756.8154296875, 896.2846069335938]
- `dog_drinking_water__petcam_20260926_173602_000_frame_3.50` (dog_drinking_water @ 3.5s): [] → ['bowl', 'cat'] [label_added, empty_to_detected, empty_to_cat, empty_to_animal]. Detections: bowl 0.791 bbox=[502.2410888671875, 417.383544921875, 930.0391845703125, 693.6995849609375]; cat 0.736 bbox=[3.9481201171875, 1.6119384765625, 828.1619873046875, 636.90625]
- `dog_drinking_water__petcam_20260926_173602_000_frame_5.50` (dog_drinking_water @ 5.5s): [] → ['bowl', 'dog'] [label_added, empty_to_detected, empty_to_dog, empty_to_animal]. Detections: bowl 0.842 bbox=[504.4384765625, 421.107177734375, 929.995361328125, 692.889892578125]; dog 0.767 bbox=[17.72747802734375, 1.60272216796875, 885.71923828125, 633.8026733398438]
- `dog_drinking_water__petcam_20260926_173602_000_frame_7.50` (dog_drinking_water @ 7.5s): ['cat'] → ['bowl'] [label_added, label_removed, animal_to_empty]. Detections: bowl 0.692 bbox=[502.9016418457031, 419.1217041015625, 930.080810546875, 692.654296875]; cat 0.417 bbox=[3.4346923828125, 1.74365234375, 643.08984375, 630.6929321289062]
- `dog_drinking_water__petcam_20260926_173602_000_frame_9.50` (dog_drinking_water @ 9.5s): ['dog'] → ['bowl', 'cat'] [label_added, label_removed, dog_to_cat]. Detections: bowl 0.740 bbox=[501.3648681640625, 429.846923828125, 931.1580810546875, 690.5133056640625]; cat 0.574 bbox=[0.47686767578125, 2.8587646484375, 699.4136352539062, 624.039306640625]; dog 0.492 bbox=[1.3331298828125, 6.88079833984375, 700.56201171875, 629.0335083007812]
- `dog_drinking_water__petcam_20260926_173602_000_frame_12.50` (dog_drinking_water @ 12.5s): ['dog'] → ['bowl', 'cat'] [label_added, label_removed, dog_to_cat]. Detections: bowl 0.742 bbox=[505.0442199707031, 421.4561767578125, 931.658203125, 690.9556884765625]; cat 0.730 bbox=[1.386474609375, 3.18829345703125, 637.1336669921875, 626.9796752929688]
- `dog_drinking_water__petcam_20260926_173602_000_frame_14.50` (dog_drinking_water @ 14.5s): ['dog'] → ['bowl', 'cat'] [label_added, label_removed, dog_to_cat]. Detections: cat 0.627 bbox=[1.4462890625, 2.626708984375, 755.0816650390625, 607.1778564453125]; bowl 0.614 bbox=[506.1101989746094, 418.4224853515625, 932.8824462890625, 688.8323974609375]
- `dog_drinking_water__petcam_20260926_173602_000_frame_30.00` (dog_drinking_water @ 30.0s): [] → ['bowl', 'dog'] [label_added, empty_to_detected, empty_to_dog, empty_to_animal]. Detections: dog 0.895 bbox=[0.0, 2.1595458984375, 883.4365844726562, 612.6598510742188]; bowl 0.755 bbox=[508.2300720214844, 409.3594665527344, 941.7432861328125, 672.9696044921875]
- `dog_drinking_water__petcam_20260926_173602_000_frame_44.00` (dog_drinking_water @ 44.0s): ['dog'] → ['cat'] [label_added, label_removed, dog_to_cat]. Detections: cat 0.643 bbox=[1.3753662109375, 1.55975341796875, 818.877685546875, 612.3010864257812]; bowl 0.476 bbox=[514.74951171875, 415.6233825683594, 942.4085693359375, 677.4468994140625]
- `dog_drinking_water__petcam_20260926_173602_000_frame_48.00` (dog_drinking_water @ 48.0s): [] → ['cat'] [label_added, empty_to_detected, empty_to_cat, empty_to_animal]. Detections: cat 0.795 bbox=[1.663330078125, 2.096435546875, 679.067138671875, 619.338134765625]; bowl 0.440 bbox=[515.5008544921875, 421.3895263671875, 943.0047607421875, 676.4898681640625]
- `dog_drinking_water__petcam_20260926_173602_000_frame_50.00` (dog_drinking_water @ 50.0s): [] → ['cat', 'chair'] [label_added, empty_to_detected, empty_to_cat, empty_to_animal]. Detections: chair 0.643 bbox=[40.0511474609375, 255.44183349609375, 245.302001953125, 492.31292724609375]; cat 0.589 bbox=[1.96368408203125, 1.67132568359375, 942.6770629882812, 611.9644165039062]
- `dog_drinking_water__petcam_20260926_173602_000_frame_56.00` (dog_drinking_water @ 56.0s): ['bowl', 'cat'] → ['bowl', 'dog'] [label_added, label_removed, cat_to_dog]. Detections: bowl 0.942 bbox=[515.25732421875, 433.27392578125, 944.197021484375, 673.973876953125]; dog 0.709 bbox=[0.90362548828125, 4.49591064453125, 909.602294921875, 632.5048828125]
- `dog_drinking_water__petcam_20260926_173602_000_frame_58.00` (dog_drinking_water @ 58.0s): [] → ['cat'] [label_added, empty_to_detected, empty_to_cat, empty_to_animal]. Detections: cat 0.830 bbox=[0.08087158203125, 2.13397216796875, 665.3374633789062, 660.9082641601562]

## 향후 canary 후보 15개 (실제 update 미수행)

유형별 후보 수: `{'cat_to_dog': 9, 'dog_to_cat': 11, 'person_removed': 2, 'person_added': 4, 'empty_to_animal': 22, 'animal_to_empty': 1, 'confidence_0.4_to_0.5': 97, 'multi_object': 192, 'unchanged': 55}`
영상 분포: `{'IMG_8450_2': 4, 'dog_drinking_water': 4, 'dog_escape': 4, 'two_dogs': 3}`

1. `IMG_8450_2__petcam_20260926_173123_000_frame_66.00` — IMG_8450_2 @ 66.0s; ['cat'] → ['bowl', 'dog']; reason: cat_to_dog, confidence_0.4_to_0.5, label_added, label_removed, multi_object; detections: dog 0.882 bbox=[850.2628784179688, 650.2279052734375, 1162.366455078125, 1004.0178833007812]; bowl 0.755 bbox=[579.7307739257812, 385.9215393066406, 742.5122680664062, 527.221923828125]; dog 0.529 bbox=[726.741943359375, 290.9590759277344, 1007.9437866210938, 661.01220703125]; cat 0.470 bbox=[727.321044921875, 291.4517822265625, 1009.4716186523438, 663.325439453125]; chair 0.423 bbox=[247.71775817871094, 2.2322845458984375, 509.7654724121094, 675.6492919921875]
2. `dog_drinking_water__petcam_20260926_173602_000_frame_12.50` — dog_drinking_water @ 12.5s; ['dog'] → ['bowl', 'cat']; reason: dog_to_cat, label_added, label_removed, multi_object; detections: bowl 0.742 bbox=[505.0442199707031, 421.4561767578125, 931.658203125, 690.9556884765625]; cat 0.730 bbox=[1.386474609375, 3.18829345703125, 637.1336669921875, 626.9796752929688]
3. `dog_drinking_water__petcam_20260926_173602_001_frame_24.00` — dog_drinking_water @ 24.0s; ['dog', 'person'] → ['bowl', 'dog']; reason: label_added, label_removed, multi_object, person_removed; detections: bowl 0.803 bbox=[514.887939453125, 419.008056640625, 945.8544921875, 674.2005615234375]; dog 0.544 bbox=[3.396240234375, 2.7890625, 869.74072265625, 632.8072509765625]
4. `dog_escape__petcam_20260926_173415_000_frame_48.00` — dog_escape @ 48.0s; [] → ['person']; reason: empty_to_detected, empty_to_person, label_added, person_added; detections: person 0.648 bbox=[600.3314819335938, 736.2921752929688, 756.8154296875, 896.2846069335938]
5. `two_dogs__two_dogs_000_frame_000005` — two_dogs @ 8.0s; [] → ['dog']; reason: confidence_0.4_to_0.5, empty_to_animal, empty_to_detected, empty_to_dog, label_added, multi_object; detections: dog 0.533 bbox=[182.466064453125, 269.7972106933594, 1164.355712890625, 943.4221801757812]; couch 0.494 bbox=[0.0, 1.438751220703125, 1674.06005859375, 1079.2435302734375]
6. `dog_drinking_water__petcam_20260926_173602_000_frame_7.50` — dog_drinking_water @ 7.5s; ['cat'] → ['bowl']; reason: animal_to_empty, confidence_0.4_to_0.5, label_added, label_removed, multi_object; detections: bowl 0.692 bbox=[502.9016418457031, 419.1217041015625, 930.080810546875, 692.654296875]; cat 0.417 bbox=[3.4346923828125, 1.74365234375, 643.08984375, 630.6929321289062]
7. `IMG_8450_2__petcam_20260926_173123_000_frame_2.00` — IMG_8450_2 @ 2.0s; ['cat'] → ['cat']; reason: confidence_0.4_to_0.5, multi_object; detections: cat 0.897 bbox=[568.7089233398438, 547.12060546875, 997.4525756835938, 769.4924926757812]; bowl 0.493 bbox=[200.76251220703125, 668.2921752929688, 377.9219055175781, 819.9747314453125]
8. `dog_escape__petcam_20260926_173415_000_frame_10.00` — dog_escape @ 10.0s; ['bed'] → ['bed']; reason: confidence_0.4_to_0.5, multi_object; detections: bed 0.740 bbox=[496.4024353027344, 76.81182861328125, 1915.375244140625, 1068.1424560546875]; couch 0.477 bbox=[502.44488525390625, 78.48028564453125, 1917.7540283203125, 1064.329833984375]
9. `two_dogs__two_dogs_000_frame_000006` — two_dogs @ 10.0s; ['couch', 'dog'] → ['couch', 'dog']; reason: multi_object; detections: dog 0.931 bbox=[219.40090942382812, 323.62957763671875, 1197.084228515625, 956.2291259765625]; couch 0.800 bbox=[0.0, 0.0966796875, 1671.7650146484375, 1080.0]
10. `IMG_8450_2__petcam_20260926_173123_000_frame_12.00` — IMG_8450_2 @ 12.0s; ['dog'] → ['bowl', 'cat']; reason: dog_to_cat, label_added, label_removed, multi_object; detections: cat 0.942 bbox=[721.8348999023438, 490.26654052734375, 1082.378662109375, 740.9039306640625]; cat 0.896 bbox=[621.8638916015625, 522.8526000976562, 955.7008666992188, 918.5953369140625]; bowl 0.618 bbox=[292.8049621582031, 638.68505859375, 467.6076965332031, 785.4948120117188]
11. `dog_escape__petcam_20260926_173415_000_frame_12.00` — dog_escape @ 12.0s; ['bed'] → ['couch']; reason: confidence_0.4_to_0.5, label_added, label_removed, multi_object; detections: couch 0.704 bbox=[637.9634399414062, 80.47897338867188, 1920.0, 530.9373779296875]; dog 0.426 bbox=[503.34228515625, 133.40341186523438, 971.5328979492188, 570.3810424804688]
12. `two_dogs__two_dogs_000_frame_000001` — two_dogs @ 0.0s; ['chair', 'tv'] → ['chair', 'couch', 'dog', 'tv']; reason: label_added, multi_object; detections: dog 0.908 bbox=[1519.0411376953125, 492.68609619140625, 1788.3189697265625, 951.162841796875]; dog 0.830 bbox=[984.309814453125, 476.1907653808594, 1104.6650390625, 734.576904296875]; tv 0.775 bbox=[373.4696044921875, 222.84100341796875, 607.6221923828125, 401.11761474609375]; chair 0.765 bbox=[1082.5235595703125, 439.0066223144531, 1337.470458984375, 869.2960205078125]; couch 0.710 bbox=[0.9863204956054688, 436.3515930175781, 224.71633911132812, 656.3037109375]; couch 0.589 bbox=[1737.19384765625, 334.62744140625, 1918.7615966796875, 1068.976806640625]
13. `IMG_8450_2__petcam_20260926_173123_000_frame_14.00` — IMG_8450_2 @ 14.0s; ['dog'] → ['bowl', 'cat']; reason: dog_to_cat, label_added, label_removed, multi_object; detections: cat 0.946 bbox=[502.5103454589844, 516.5958251953125, 975.2345581054688, 899.35107421875]; cat 0.862 bbox=[718.8256225585938, 460.3310241699219, 1072.90380859375, 725.3113403320312]; bowl 0.859 bbox=[287.110107421875, 617.965087890625, 461.66619873046875, 766.9610595703125]
14. `dog_drinking_water__petcam_20260926_173602_000_frame_14.50` — dog_drinking_water @ 14.5s; ['dog'] → ['bowl', 'cat']; reason: dog_to_cat, label_added, label_removed, multi_object; detections: cat 0.627 bbox=[1.4462890625, 2.626708984375, 755.0816650390625, 607.1778564453125]; bowl 0.614 bbox=[506.1101989746094, 418.4224853515625, 932.8824462890625, 688.8323974609375]
15. `dog_escape__petcam_20260926_173415_000_frame_14.00` — dog_escape @ 14.0s; ['bed'] → ['couch']; reason: confidence_0.4_to_0.5, label_added, label_removed, multi_object; detections: couch 0.526 bbox=[363.63043212890625, 126.75631713867188, 1912.9456787109375, 1071.818115234375]; bed 0.415 bbox=[330.97723388671875, 129.08880615234375, 1906.0660400390625, 1069.404052734375]
