# ML Challenge: Fault Detection in Device Activity

## Project Description
This project builds a classification model to predict fault conditions in devices based on 47 numerical sensor features. The target variable `Class` indicates whether a device is operating normally (0) or exhibiting a fault (1).

## Dataset
- **TRAIN.csv** – 43,776 samples with 47 numerical features (F01–F47) and target `Class`
- **TEST.csv** – 10,944 samples with 47 features and an `ID` column (no `Class`)

## Approach

### Feature Engineering
Raw features were kept as-is (no missing values, no constant features). Additionally, 10 row-wise aggregate features were derived to capture the overall device state across all sensors:

| Feature | Description |
|---|---|
| `mean`, `std`, `median` | Signal level and spread |
| `max`, `min`, `range` | Extremes and dynamic range |
| `energy` | Sum of squared values — faulty devices tend to show higher energy |
| `q75_q25` | IQR — robust spread measure |
| `sum_neg` | Count of negative readings |
| `cv` | Coefficient of variation (std/mean) |

The idea is that faults usually don't show up in a single sensor — they affect multiple sensors at once, so row-level statistics help the model pick that up.

### Model
Used `HistGradientBoostingClassifier` from sklearn — it's sklearn's fast gradient boosting implementation, similar to XGBoost. Performed better than RandomForest and ensemble combinations in cross-validation.

Key hyperparameters:
```
learning_rate = 0.1
max_depth = 7
min_samples_leaf = 10
max_iter = 1000
early_stopping = True
```

### Training Strategy
- **5-Fold Stratified Cross-Validation** to preserve class distribution across folds
- **Out-of-Fold (OOF) probabilities** collected across all folds
- **Test predictions averaged** across all 5 fold models (reduces variance)
- **Threshold tuning** — instead of default 0.5, searched for the threshold that maximizes F1 on OOF predictions (found: ~0.315)

## Performance

Gives  98.9587 out of 100 on the evaluation engine provided by ieee team.

## Running the Notebook

1. Place `TRAIN.csv` and `TEST.csv` in the same directory as the notebook
2. Run all cells sequentially — the notebook covers:
   - Data loading and exploration
   - Feature engineering
   - Model training with 5-fold CV
   - Threshold optimization
   - Performance metrics and confusion matrix
   - Saving `FINAL.csv`

## Output

`FINAL.csv` with two columns — `ID` and `CLASS` — containing predictions for all 10,944 test entries.
