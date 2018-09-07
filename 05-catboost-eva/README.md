## Evaluation of catboost on MCA with nested cross-validation

### Code

`find_best_params.py`: code to search for best params on all CV splits and dump results as csv

`train_predict.py`: code to train model with best parameters from the previous step and predict the test part for each CV split. Predictions are stored in `cv*-predictions.csv`

`dump_cv_idx.py`: code to dump CV splits as cell ids. Dumps to `cv-idx` folder

`seurat_classify_cells.R`: code to run Seurat's `ClassifyCells` random forest algorithm on CV splits from the previous steps. Predictios are stored in `cv*-preds-seurat.csv`

`score.py`: code to read Seurat and catboost prediction from previous steps and score them. Final table is in `scores.csv`

`cv_confusion_matrix.py`: code to plot confusion matrix for CV run #3. Figure 3

### Cached files

`cv*-predictions.csv`: best catboost predictions for each CV run

`scores.csv`: scores for Seurat and catboost performance. Table 1
