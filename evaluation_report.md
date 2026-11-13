# Plant Disease Classifier Evaluation Report

| Metric | Result |
| --- | ---: |
| Overall accuracy | 97.84% |
| Balanced accuracy | 97.25% |

## Calculation

Both metrics were calculated from predictions on 10,709 test images across 38 classes.

- **Overall accuracy:** correct predictions divided by all test images, multiplied by 100. Each image has equal weight.
- **Balanced accuracy:** calculate recall for each class (correct predictions in that class divided by the number of test images in that class), then average the 38 recalls and multiply by 100. Each class has equal weight.

The reported percentages are rounded to two decimal places.
