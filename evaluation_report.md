# Plant Disease Classifier Evaluation Report

## Summary

The saved ResNet-50 checkpoint was evaluated on all 10,709 images listed in the project's PlantVillage `color_test.txt` split. The split contains 38 classes, and all listed image files were present.

| Metric | Result |
| --- | ---: |
| Accuracy (overall) | 97.84% |
| Balanced accuracy | 97.25% |
| Macro precision | 97.49% |
| Macro recall | 97.25% |
| Macro F1-score | 97.25% |
| Support-weighted precision | 97.99% |
| Support-weighted recall | 97.84% |
| Support-weighted F1-score | 97.82% |
| Macro one-vs-rest ROC-AUC | 0.999877 |
| Support-weighted one-vs-rest ROC-AUC | 0.999895 |

Macro scores give every class equal weight. Support-weighted scores weight each class by its number of test images. Overall accuracy and weighted recall are equal for this single-label multiclass task.

## Evaluation Method

The checkpoint was loaded with the project's checkpoint validation and model-loading functions. Test labels were read from each image's class directory and mapped using the class order stored in the checkpoint. Each image was converted to RGB, resized to 224 x 224 pixels, converted to a tensor, and normalized with the ImageNet mean and standard deviation configured by the project. Test-time augmentation was not used. Predicted classes were the class with the highest output logit. ROC-AUC was calculated from the softmax probability for each class. Inference ran on CPU in batches of 32.

A 38 x 38 confusion matrix was accumulated, where rows are true classes and columns are predicted classes. For class `c`, `TP_c` is its diagonal count, `FP_c` is the rest of its prediction column, and `FN_c` is the rest of its true-label row.

### Metric Definitions

- **Accuracy:** total correct predictions divided by the total number of images: `sum(TP_c) / N`. This weights classes according to their frequency in the split.
- **Balanced accuracy:** the arithmetic mean of per-class recall: `mean_c(TP_c / (TP_c + FN_c))`. In this single-label multiclass evaluation, it is the same as macro recall and gives each class equal weight.
- **Precision:** per-class precision is `TP_c / (TP_c + FP_c)`, the fraction of predictions for a class that are correct. The reported macro precision is the unweighted mean across classes; weighted precision weights each class by its true support.
- **Recall:** per-class recall is `TP_c / (TP_c + FN_c)`, the fraction of images of a class correctly identified. Macro recall is the unweighted class mean; weighted recall uses class support.
- **F1-score:** per-class F1 is the harmonic mean of precision and recall: `2 * precision_c * recall_c / (precision_c + recall_c)`. Macro F1 averages the 38 class F1-scores; weighted F1 averages them using true class support.
- **ROC-AUC:** for each class, a one-vs-rest ROC curve was formed by varying the class-probability threshold and plotting true-positive rate against false-positive rate. Its area was calculated with trapezoidal integration, grouping equal scores at the same threshold. Macro ROC-AUC is the unweighted mean of the 38 class AUCs; support-weighted ROC-AUC weights each class by its number of true examples. ROC-AUC evaluates ranking over thresholds, unlike accuracy, precision, recall, and F1, which use the top-scoring predicted class.

When a class has no predicted samples, its precision is defined as zero; when precision and recall sum to zero, its F1-score is defined as zero.

## Per-Class Results

| Class | Support | Precision | Recall | F1-score | One-vs-rest ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Apple___Apple_scab | 127 | 99.20% | 97.64% | 98.41% | 0.999989 |
| Apple___Black_rot | 146 | 100.00% | 100.00% | 100.00% | 1.000000 |
| Apple___Cedar_apple_rust | 52 | 100.00% | 100.00% | 100.00% | 1.000000 |
| Apple___healthy | 286 | 99.65% | 98.25% | 98.94% | 0.999993 |
| Blueberry___healthy | 324 | 99.08% | 99.69% | 99.38% | 0.999991 |
| Cherry_(including_sour)___Powdery_mildew | 219 | 99.10% | 100.00% | 99.55% | 1.000000 |
| Cherry_(including_sour)___healthy | 123 | 99.19% | 100.00% | 99.60% | 1.000000 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 107 | 78.29% | 94.39% | 85.59% | 0.999498 |
| Corn_(maize)___Common_rust_ | 217 | 100.00% | 100.00% | 100.00% | 1.000000 |
| Corn_(maize)___Northern_Leaf_Blight | 220 | 96.46% | 86.82% | 91.39% | 0.999727 |
| Corn_(maize)___healthy | 250 | 99.60% | 100.00% | 99.80% | 1.000000 |
| Grape___Black_rot | 168 | 95.38% | 98.21% | 96.77% | 0.999980 |
| Grape___Esca_(Black_Measles) | 284 | 99.30% | 100.00% | 99.65% | 1.000000 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 204 | 100.00% | 99.51% | 99.75% | 1.000000 |
| Grape___healthy | 100 | 100.00% | 100.00% | 100.00% | 1.000000 |
| Orange___Haunglongbing_(Citrus_greening) | 980 | 99.80% | 100.00% | 99.90% | 1.000000 |
| Peach___Bacterial_spot | 542 | 99.45% | 100.00% | 99.72% | 1.000000 |
| Peach___healthy | 71 | 98.59% | 98.59% | 98.59% | 0.999997 |
| Pepper,_bell___Bacterial_spot | 166 | 99.40% | 99.40% | 99.40% | 0.999997 |
| Pepper,_bell___healthy | 283 | 99.30% | 99.65% | 99.47% | 0.999944 |
| Potato___Early_blight | 164 | 85.42% | 100.00% | 92.13% | 0.999969 |
| Potato___Late_blight | 212 | 99.00% | 93.87% | 96.37% | 0.999919 |
| Potato___healthy | 32 | 100.00% | 96.88% | 98.41% | 0.999997 |
| Raspberry___healthy | 162 | 100.00% | 87.04% | 93.07% | 0.999987 |
| Soybean___healthy | 973 | 98.98% | 99.79% | 99.39% | 0.999993 |
| Squash___Powdery_mildew | 378 | 100.00% | 100.00% | 100.00% | 1.000000 |
| Strawberry___Leaf_scorch | 200 | 100.00% | 99.50% | 99.75% | 1.000000 |
| Strawberry___healthy | 80 | 98.77% | 100.00% | 99.38% | 1.000000 |
| Tomato___Bacterial_spot | 436 | 99.08% | 98.62% | 98.85% | 0.999975 |
| Tomato___Early_blight | 215 | 94.77% | 75.81% | 84.24% | 0.998276 |
| Tomato___Late_blight | 375 | 93.25% | 99.47% | 96.26% | 0.999925 |
| Tomato___Leaf_Mold | 172 | 98.27% | 98.84% | 98.55% | 0.999972 |
| Tomato___Septoria_leaf_spot | 337 | 98.78% | 96.14% | 97.44% | 0.999923 |
| Tomato___Spider_mites Two-spotted_spider_mite | 320 | 83.33% | 100.00% | 90.91% | 0.999735 |
| Tomato___Target_Spot | 288 | 93.36% | 78.13% | 85.07% | 0.998537 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 1100 | 100.00% | 99.82% | 99.91% | 0.999999 |
| Tomato___Tomato_mosaic_virus | 66 | 100.00% | 100.00% | 100.00% | 1.000000 |
| Tomato___healthy | 330 | 100.00% | 99.39% | 99.70% | 0.999999 |

## Model and Data Details

- Dataset: PlantVillage, color images, project-pinned revision `9e97599868962bd0079b8db4b7f1efa9185fa1e7`.
- Split: `data/plantvillage/splits/color_test.txt` (10,709 images, 38 classes).
- Checkpoint: `plant_disease_resnet50.pth`, saved at epoch 1; its recorded validation accuracy is 97.84%.
- Runtime: CPU; evaluation used the project's ResNet-50 and inference preprocessing.

## Interpretation and Limitation

The project currently evaluates this `color_test.txt` split during training and uses its accuracy to select and save the checkpoint. Therefore, these reported results describe performance on the split used for checkpoint selection; they are not an independent, untouched final-test estimate and may be optimistically biased. A separate holdout set is needed for an unbiased final evaluation. PlantVillage images are also captured in controlled conditions, so these results do not establish accuracy on field photographs or provide a confirmed plant-disease diagnosis.