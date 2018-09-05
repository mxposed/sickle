## Clustering of SC01, SC02 and SC03 with Seurat

##### Code

`main.R`: code to initially cluster SC01, SC02 and SC03. Caches Seurat objects in RDS files, saves PCA elbow plots, tSNE plots and tables with marker genes for each cluster.

`sc01.R`: code to produce final clustering of SC01 (SC01v2).

`sc02.R`: code to produce final clustering of SC02 (SC02v2).

`sc02_b_cells.R`: code to inspect differences between 2 clusters of B cells in the initial clustering of SC02.

`plot_sc02_10.R`: code to plot Figure 2: SC02v2 tSNE + expression of DC subtypes marker genes in cluster #10.

##### Cached files

`*_markers_*.csv`: tables of genes, that characterise each cluster in each dataset. This is the output of `FindAllMarkers` function in Seurat. It lists top differentially-expressed genes for all clusters, with their average log-fold change and correspondent p-value. For cluster names see `00-metadata`. For this table for MCA dataset, see `13-cluster-mca`.

`*_assgn.csv`: tables of cell assignments in datasets. Contains cell ids and their cluster number.
