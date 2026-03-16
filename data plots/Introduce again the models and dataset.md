Introduce again the models and dataset 
-metrics used to evaluate
-comparative table of the accuracies , Macro-F1, and Cohen’s Kappa, Precision, Recall for all the models
-having 5 figure comparing the accuracies and losses of the different models

================================================================================
MODEL EVALUATION for DensNet121
================================================================================
Generating predictions...

--------------------------------------------------------------------------------
OVERALL METRICS
--------------------------------------------------------------------------------
Test Accuracy:  0.9378 (93.78%)
Test Loss:      0.1678
Weighted F1:    0.9371
Macro F1:       0.9363
MCC:            0.9225
Cohen's Kappa:  0.9222

--------------------------------------------------------------------------------
CLASSIFICATION REPORT
--------------------------------------------------------------------------------
              precision    recall  f1-score   support

       0 (1)     1.0000    1.0000    1.0000        84
       1 (1)     0.9231    1.0000    0.9600        84
       2 (1)     0.9014    0.8533    0.8767        75
       3 (1)     0.9136    0.8706    0.8916        85
       4 (1)     0.9467    0.9595    0.9530        74

    accuracy                         0.9378       402
   macro avg     0.9369    0.9367    0.9363       402
weighted avg     0.9374    0.9378    0.9371       402

--------------------------------------------------------------------------------
ROC-AUC & PR-AUC SCORES
--------------------------------------------------------------------------------
0 (1)      - ROC-AUC: 1.0000, PR-AUC: 1.0000
1 (1)      - ROC-AUC: 0.9985, PR-AUC: 0.9941
2 (1)      - ROC-AUC: 0.9886, PR-AUC: 0.9584
3 (1)      - ROC-AUC: 0.9920, PR-AUC: 0.9685
4 (1)      - ROC-AUC: 0.9988, PR-AUC: 0.9953

Macro Average - ROC-AUC: 0.9956, PR-AUC: 0.9833

✓ Results saved to: /kaggle/working/DenseNet121_20251220_103749/evaluation_results.json

Generating confusion matrix plots...
✓ Saved: fig2_confusion_matrix.png
Generating ROC curves...
✓ Saved: fig3_roc_curves.png
Generating Precision-Recall curves...
✓ Saved: fig4_precision_recall_curves.png
Generating per-class metrics plot...
✓ Saved: fig5_per_class_metrics.png

✓ Final model saved to: /kaggle/working/DenseNet121_20251220_103749/models/final_model.keras

================================================================================
PIPELINE COMPLETED SUCCESSFULLY
================================================================================

All outputs saved to: /kaggle/working/DenseNet121_20251220_103749

Final Test Accuracy: 0.9378
Macro F1-Score: 0.9363
Cohen's Kappa: 0.9222

================================================================================


================================================================================
MODEL EVALUATION for EfficientNetB0
================================================================================
Generating predictions...

--------------------------------------------------------------------------------
OVERALL METRICS
--------------------------------------------------------------------------------
Test Accuracy:  0.4726 (47.26%)
Test Loss:      1.2312
Weighted F1:    0.4281
Macro F1:       0.4317
MCC:            0.3549
Cohen's Kappa:  0.3389

--------------------------------------------------------------------------------
CLASSIFICATION REPORT
--------------------------------------------------------------------------------
              precision    recall  f1-score   support

       0 (1)     0.4724    0.7143    0.5687        84
       1 (1)     0.4247    0.7381    0.5391        84
       2 (1)     0.4737    0.3600    0.4091        75
       3 (1)     0.2857    0.0471    0.0808        85
       4 (1)     0.6379    0.5000    0.5606        74

    accuracy                         0.4726       402
   macro avg     0.4589    0.4719    0.4317       402
weighted avg     0.4537    0.4726    0.4281       402

--------------------------------------------------------------------------------
ROC-AUC & PR-AUC SCORES
--------------------------------------------------------------------------------
0 (1)      - ROC-AUC: 0.8190, PR-AUC: 0.6128
1 (1)      - ROC-AUC: 0.8427, PR-AUC: 0.4948
2 (1)      - ROC-AUC: 0.8519, PR-AUC: 0.5867
3 (1)      - ROC-AUC: 0.6957, PR-AUC: 0.2894
4 (1)      - ROC-AUC: 0.8568, PR-AUC: 0.6734

Macro Average - ROC-AUC: 0.8132, PR-AUC: 0.5314

✓ Results saved to: /kaggle/working/EfficientNetB0_20251220_105111/evaluation_results.json

Generating confusion matrix plots...
✓ Saved: fig2_confusion_matrix.png
Generating ROC curves...
✓ Saved: fig3_roc_curves.png
Generating Precision-Recall curves...
✓ Saved: fig4_precision_recall_curves.png
Generating per-class metrics plot...
✓ Saved: fig5_per_class_metrics.png

✓ Final model saved to: /kaggle/working/EfficientNetB0_20251220_105111/models/final_model.keras

================================================================================
PIPELINE COMPLETED SUCCESSFULLY
================================================================================

All outputs saved to: /kaggle/working/EfficientNetB0_20251220_105111

Final Test Accuracy: 0.4726
Macro F1-Score: 0.4317
Cohen's Kappa: 0.3389

================================================================================

================================================================================
MODEL EVALUATION for InceptionResNetV2
================================================================================
Generating predictions...

--------------------------------------------------------------------------------
OVERALL METRICS
--------------------------------------------------------------------------------
Test Accuracy:  0.9328 (93.28%)
Test Loss:      0.1304
Weighted F1:    0.9320
Macro F1:       0.9311
MCC:            0.9163
Cohen's Kappa:  0.9159

--------------------------------------------------------------------------------
CLASSIFICATION REPORT
--------------------------------------------------------------------------------
              precision    recall  f1-score   support

       0 (1)     1.0000    1.0000    1.0000        84
       1 (1)     0.9121    0.9881    0.9486        84
       2 (1)     0.8971    0.8133    0.8531        75
       3 (1)     0.8941    0.8941    0.8941        85
       4 (1)     0.9595    0.9595    0.9595        74

    accuracy                         0.9328       402
   macro avg     0.9325    0.9310    0.9311       402
weighted avg     0.9326    0.9328    0.9320       402

--------------------------------------------------------------------------------
ROC-AUC & PR-AUC SCORES
--------------------------------------------------------------------------------
0 (1)      - ROC-AUC: 1.0000, PR-AUC: 1.0000
1 (1)      - ROC-AUC: 0.9984, PR-AUC: 0.9938
2 (1)      - ROC-AUC: 0.9901, PR-AUC: 0.9634
3 (1)      - ROC-AUC: 0.9913, PR-AUC: 0.9647
4 (1)      - ROC-AUC: 0.9988, PR-AUC: 0.9954

Macro Average - ROC-AUC: 0.9957, PR-AUC: 0.9835

✓ Results saved to: /kaggle/working/InceptionResNetV2_20251220_112247/evaluation_results.json

Generating confusion matrix plots...
✓ Saved: fig2_confusion_matrix.png
Generating ROC curves...
✓ Saved: fig3_roc_curves.png
Generating Precision-Recall curves...
✓ Saved: fig4_precision_recall_curves.png
Generating per-class metrics plot...
✓ Saved: fig5_per_class_metrics.png

✓ Final model saved to: /kaggle/working/InceptionResNetV2_20251220_112247/models/final_model.keras

================================================================================
PIPELINE COMPLETED SUCCESSFULLY
================================================================================

All outputs saved to: /kaggle/working/InceptionResNetV2_20251220_112247

Final Test Accuracy: 0.9328
Macro F1-Score: 0.9311
Cohen's Kappa: 0.9159

================================================================================


================================================================================
MODEL EVALUATION for InceptionV3
================================================================================
Generating predictions...

--------------------------------------------------------------------------------
OVERALL METRICS
--------------------------------------------------------------------------------
Test Accuracy:  0.9179 (91.79%)
Test Loss:      0.2325
Weighted F1:    0.9165
Macro F1:       0.9153
MCC:            0.8980
Cohen's Kappa:  0.8972

--------------------------------------------------------------------------------
CLASSIFICATION REPORT
--------------------------------------------------------------------------------
              precision    recall  f1-score   support

       0 (1)     1.0000    1.0000    1.0000        84
       1 (1)     0.9032    1.0000    0.9492        84
       2 (1)     0.9219    0.7867    0.8489        75
       3 (1)     0.8675    0.8471    0.8571        85
       4 (1)     0.8974    0.9459    0.9211        74

    accuracy                         0.9179       402
   macro avg     0.9180    0.9159    0.9153       402
weighted avg     0.9183    0.9179    0.9165       402

--------------------------------------------------------------------------------
ROC-AUC & PR-AUC SCORES
--------------------------------------------------------------------------------
0 (1)      - ROC-AUC: 1.0000, PR-AUC: 1.0000
1 (1)      - ROC-AUC: 0.9935, PR-AUC: 0.9742
2 (1)      - ROC-AUC: 0.9793, PR-AUC: 0.9358
3 (1)      - ROC-AUC: 0.9855, PR-AUC: 0.9457
4 (1)      - ROC-AUC: 0.9972, PR-AUC: 0.9888

Macro Average - ROC-AUC: 0.9911, PR-AUC: 0.9689

✓ Results saved to: /kaggle/working/InceptionV3_20251220_120548/evaluation_results.json

Generating confusion matrix plots...
✓ Saved: fig2_confusion_matrix.png
Generating ROC curves...
✓ Saved: fig3_roc_curves.png
Generating Precision-Recall curves...
✓ Saved: fig4_precision_recall_curves.png
Generating per-class metrics plot...
✓ Saved: fig5_per_class_metrics.png

✓ Final model saved to: /kaggle/working/InceptionV3_20251220_120548/models/final_model.keras

================================================================================
PIPELINE COMPLETED SUCCESSFULLY
================================================================================

All outputs saved to: /kaggle/working/InceptionV3_20251220_120548

Final Test Accuracy: 0.9179
Macro F1-Score: 0.9153
Cohen's Kappa: 0.8972

================================================================================

================================================================================
MODEL EVALUATION MobileNetV3Small
================================================================================
Generating predictions...

--------------------------------------------------------------------------------
OVERALL METRICS
--------------------------------------------------------------------------------
Test Accuracy:  0.7313 (73.13%)
Test Loss:      0.6382
Weighted F1:    0.7324
Macro F1:       0.7339
MCC:            0.6700
Cohen's Kappa:  0.6646

--------------------------------------------------------------------------------
CLASSIFICATION REPORT
--------------------------------------------------------------------------------
              precision    recall  f1-score   support

       0 (1)     0.9429    0.7857    0.8571        84
       1 (1)     0.7500    0.5714    0.6486        84
       2 (1)     0.6018    0.9067    0.7234        75
       3 (1)     0.6395    0.6471    0.6433        85
       4 (1)     0.8261    0.7703    0.7972        74

    accuracy                         0.7313       402
   macro avg     0.7520    0.7362    0.7339       402
weighted avg     0.7533    0.7313    0.7324       402

--------------------------------------------------------------------------------
ROC-AUC & PR-AUC SCORES
--------------------------------------------------------------------------------
0 (1)      - ROC-AUC: 0.9805, PR-AUC: 0.9408
1 (1)      - ROC-AUC: 0.9304, PR-AUC: 0.8060
2 (1)      - ROC-AUC: 0.9649, PR-AUC: 0.8860
3 (1)      - ROC-AUC: 0.9124, PR-AUC: 0.7127
4 (1)      - ROC-AUC: 0.9659, PR-AUC: 0.8998

Macro Average - ROC-AUC: 0.9508, PR-AUC: 0.8491

✓ Results saved to: /kaggle/working/MobileNetV3Small_20251220_123859/evaluation_results.json

Generating confusion matrix plots...
✓ Saved: fig2_confusion_matrix.png
Generating ROC curves...
✓ Saved: fig3_roc_curves.png
Generating Precision-Recall curves...
✓ Saved: fig4_precision_recall_curves.png
Generating per-class metrics plot...
✓ Saved: fig5_per_class_metrics.png

✓ Final model saved to: /kaggle/working/MobileNetV3Small_20251220_123859/models/final_model.keras

================================================================================
PIPELINE COMPLETED SUCCESSFULLY
================================================================================

All outputs saved to: /kaggle/working/MobileNetV3Small_20251220_123859

Final Test Accuracy: 0.7313
Macro F1-Score: 0.7339
Cohen's Kappa: 0.6646

================================================================================
