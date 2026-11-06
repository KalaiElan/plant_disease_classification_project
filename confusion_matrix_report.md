# Plant Disease Classifier Confusion Matrix

## Evaluation Setup

This report gives one-vs-rest confusion counts for each of the 38 classes, computed from the model's predictions on all 10,709 images in `data/plantvillage/splits/color_test.txt`. Each row treats the named class as positive and all other classes combined as negative. The evaluation used `plant_disease_resnet50.pth`, the project's inference preprocessing, and CPU inference.

## Count Definitions

- **True Positive (TP):** an image belongs to this class and is predicted as this class.
- **True Negative (TN):** an image does not belong to this class and is not predicted as this class.
- **False Positive (FP):** an image belongs to another class but is predicted as this class.
- **False Negative (FN):** an image belongs to this class but is predicted as another class.

Each row satisfies `TP + TN + FP + FN = 10,709`. For each class, the counts come directly from the multiclass confusion matrix: TP is its diagonal entry; FN is the rest of its true-class row; FP is the rest of its predicted-class column; TN is the total number of test images minus TP, FN, and FP.

## Per-Class Counts

| Class | TP | TN | FP | FN |
| --- | ---: | ---: | ---: | ---: |
| Apple___Apple_scab | 124 | 10581 | 1 | 3 |
| Apple___Black_rot | 146 | 10563 | 0 | 0 |
| Apple___Cedar_apple_rust | 52 | 10657 | 0 | 0 |
| Apple___healthy | 281 | 10422 | 1 | 5 |
| Blueberry___healthy | 323 | 10382 | 3 | 1 |
| Cherry_(including_sour)___Powdery_mildew | 219 | 10488 | 2 | 0 |
| Cherry_(including_sour)___healthy | 123 | 10585 | 1 | 0 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 101 | 10574 | 28 | 6 |
| Corn_(maize)___Common_rust_ | 217 | 10492 | 0 | 0 |
| Corn_(maize)___Northern_Leaf_Blight | 191 | 10482 | 7 | 29 |
| Corn_(maize)___healthy | 250 | 10458 | 1 | 0 |
| Grape___Black_rot | 165 | 10533 | 8 | 3 |
| Grape___Esca_(Black_Measles) | 284 | 10423 | 2 | 0 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 203 | 10505 | 0 | 1 |
| Grape___healthy | 100 | 10609 | 0 | 0 |
| Orange___Haunglongbing_(Citrus_greening) | 980 | 9727 | 2 | 0 |
| Peach___Bacterial_spot | 542 | 10164 | 3 | 0 |
| Peach___healthy | 70 | 10637 | 1 | 1 |
| Pepper,_bell___Bacterial_spot | 165 | 10542 | 1 | 1 |
| Pepper,_bell___healthy | 282 | 10424 | 2 | 1 |
| Potato___Early_blight | 164 | 10517 | 28 | 0 |
| Potato___Late_blight | 199 | 10495 | 2 | 13 |
| Potato___healthy | 31 | 10677 | 0 | 1 |
| Raspberry___healthy | 141 | 10547 | 0 | 21 |
| Soybean___healthy | 971 | 9726 | 10 | 2 |
| Squash___Powdery_mildew | 378 | 10331 | 0 | 0 |
| Strawberry___Leaf_scorch | 199 | 10509 | 0 | 1 |
| Strawberry___healthy | 80 | 10628 | 1 | 0 |
| Tomato___Bacterial_spot | 430 | 10269 | 4 | 6 |
| Tomato___Early_blight | 163 | 10485 | 9 | 52 |
| Tomato___Late_blight | 373 | 10307 | 27 | 2 |
| Tomato___Leaf_Mold | 170 | 10534 | 3 | 2 |
| Tomato___Septoria_leaf_spot | 324 | 10368 | 4 | 13 |
| Tomato___Spider_mites Two-spotted_spider_mite | 320 | 10325 | 64 | 0 |
| Tomato___Target_Spot | 225 | 10405 | 16 | 63 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 1098 | 9609 | 0 | 2 |
| Tomato___Tomato_mosaic_virus | 66 | 10643 | 0 | 0 |
| Tomato___healthy | 328 | 10379 | 0 | 2 |

## Interpretation Note

The test split is also used during training to select the saved checkpoint, so the counts describe performance on the checkpoint-selection split rather than an independent, untouched final test set. See `evaluation_report.md` for aggregate metrics and additional evaluation limitations.