# Fast and consistent annotation of scRNA-seq experiments by predicting novel and reference cell types with catboost
Code for M.Sc. thesis project

#### Dependencies
Code is written in python 3 and R. Python requirements are in `requirements.txt`, R requirements are as follow:
  * Seurat
  * scmap
  * scran
  * ggplot2
  * MUDAN (optional; https://jef.works/MUDAN/)
  * SeuratConverter
  * irr

Datasets are expected to be in the parent folder of this code root, like this:

`<folder>/`
  * `sickle/` this repo
  * `rmbatch_dge/` MCA data with Lung batches unzipped
  * `SC01/` Reyfman _et al._ samples
  * `SC02/`
  * `SC03/`

#### General notes
This repo contains several data files in `00-metadata`, and other data files are cached after they were created by scripts, and can be reproduced from datasets.

Python scripts are expected to be run with `run.sh <script>` (it adds `lib` to PYTHONPATH).

#### Code structure
`lib`: common code extracted from python scripts. Includes loading datasets, predictions, mapping and quantifying cross-dataset predictions, drawing Sankey diagrams.

`00-metadata`: data files with cluster names and cluster correspondence between datasets

`01-cluster-sc01-sc02`: code to cluster SC01, SC02 and SC03 datasets and cache them. Code for Figure 2. Code to inspect _B cells_ split in initial clustering of SC02

`02-evaluate-mnn`: code to test MNN batch correction between SC01 and SC03. Haghverdi,L. et al. (2018) Batch effects in single-cell RNA-sequencing data are corrected by matching mutual nearest neighbors. Nat. Biotechnol., 36, 421–427.

`03-evaluate-seurat`: code to test CCA-based batch correction approach from Seurat between SC01 and SC03. Butler,A. et al. (2018) Integrating single-cell transcriptomic data across different conditions, technologies, and species. Nat. Biotechnol., 36, 411–420.

`04-evaluate-scmap`: code to test scmap projection method. Code to quantify and plot SC03 projection onto SC02 dataset with scmap-cluster method, Figure 10 (left part). Kiselev,V.Y. et al. (2018) Scmap: Projection of single-cell RNA-seq data across data sets. Nat. Methods, 15, 359–362.

`05-catboost-eva`: evaluation of catboost on MCA dataset using nested-cv. Code to get baseline predictions. Plotting of Figure 3.

`06-select-measure`: code to test how different metrics and measures for clustering comparison react to different cluster perturbations. Sketch of custom measure “mapScore”

`07-catboost-sc02`: code to select and train best catboost model for SC02, predict SC01 and SC03 with it, plot and quantify. Figure 4 (left part). Figure 8.

`08-unseen-sc02`: code to test several ensemble models trained on SC02, predict SC03, quantify and plot predictions. Figure 9. Figure 10 (right part).

`09-leave-cell-type-out`: code to run leave-one-cluster-out cross-validation for ensemble method, and to search for the best threshold for “novel cell type” detection on SC02.

`10-catboost-sc01`: code to select and train best model for SC01, predict, plot and quantify SC02. Figure 4 (right part).

`11-catboost-sc03`: code to select and train best model for SC03; predict SC01 and SC02, plot and quantify; plot cells predicted as _Plasma cells_. Figure 7.

`12-catboost-mca`: code to select and train best model for MCA; predict SC01, SC02 and SC03, plot and quantify; run correlation analysis and plot correlation plots.

`13-cluster-mca`: cell type assignment table for our clustering of MCA Lung dataset. Clustering itself was done by Dr. Misharin (https://osf.io/agc98/)
