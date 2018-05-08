require(scmap)
require(SingleCellExperiment)
require(SeuratConverter)
require(irr)

CURRENT_DIR <- dirname(sys.frame(1)$ofile)
CODE_ROOT <- dirname(CURRENT_DIR)
SCE_CACHE_DIR <- file.path(CODE_ROOT, '01-cluster-sc01-sc02')
METADATA_DIR <- file.path(CODE_ROOT, '00-metadata')

getScmapScore <- function(ref, query) {
  # genes <- head(rownames(sc01.seurat@hvg.info), no_of_genes)
  # rowData(sc01)$scmap_features <- rownames(sc01) %in% genes
  ref <- indexCluster(ref)
  scmap_result <- scmapCluster(projection=query, 
                               index_list=list(
                                 ref=metadata(ref)$scmap_cluster_index
                               ))
  kappa <- kappa2(data.frame(
    colData(query)$cell_type1,
    scmap_result$combined_labs
  )[scmap_result$combined_labs != "unassigned", ])$value
  assigned_frac <- sum(scmap_result$combined_labs != "unassigned") / ncol(query)
  return(list(kappa=kappa, assigned_frac=assigned_frac))
}

main <- function() {
  sc01.seurat <- readRDS(file.path(SCE_CACHE_DIR, 'SC01.rds'))
  sc01.clusters <- read.csv(file.path(METADATA_DIR, 'SC01_clusters.csv'), header=FALSE)
  sc01.clusters$V1 <- sub("C", "", sc01.clusters$V1)
  sc01 <- as(sc01.seurat, "SingleCellExperiment")
  counts(sc01) <- as.matrix(assay(sc01, "raw.data"))
  logcounts(sc01) <- log2(counts(sc01) + 1)
  rowData(sc01)$feature_symbol <- rownames(sc01)
  colData(sc01)$cell_type1 <- sc01.seurat@ident
  
  rowData(sc01)$scmap_features <- rownames(sc01) %in% rownames(head(sc01.seurat@hvg.info, 500))
  res <- getScmapScore(sc01, sc01)
  cat(paste("Scmap SC01 on SC01.
counts <- raw.data
logcounts <- log2(counts + 1)
Seurat top 500 features
Kappa:", res$kappa, ", Assigned frac:", res$assigned_frac),
      file=file.path(CURRENT_DIR, "sc01-on-sc01.txt"))

  sc01 <- selectFeatures(sc01)
  res <- getScmapScore(sc01, sc01)
  cat(paste("\n\nScmap SC01 on SC01.
counts <- raw.data
logcounts <- log2(counts + 1)
Kappa:", res$kappa, ", Assigned frac:", res$assigned_frac),
      file=file.path(CURRENT_DIR, "sc01-on-sc01.txt"), append=TRUE)
  
  counts(sc01) <- as.matrix(assay(sc01, "raw.data"))
  logcounts(sc01) <- as.matrix(assay(sc01, "data"))
  sc01 <- selectFeatures(sc01)
  res <- getScmapScore(sc01, sc01)
  cat(paste("\n\nScmap SC01 on SC01.
counts <- raw.data
logcounts <- data
Kappa:", res$kappa, ", Assigned frac:", res$assigned_frac),
      file=file.path(CURRENT_DIR, "sc01-on-sc01.txt"), append=TRUE)
  
  counts(sc01) <- NULL
  logcounts(sc01) <- as.matrix(assay(sc01, "data"))
  sc01 <- selectFeatures(sc01)
  res <- getScmapScore(sc01, sc01)
  cat(paste("\n\nScmap SC01 on SC01.
counts <- NULL
logcounts <- data
Kappa:", res$kappa, ", Assigned frac:", res$assigned_frac),
      file=file.path(CURRENT_DIR, "sc01-on-sc01.txt"), append=TRUE)
  
  sc01 <- as(sc01.seurat, "SingleCellExperiment")
  counts(sc01) <- as.matrix(assay(sc01, "raw.data"))
  logcounts(sc01) <- log2(counts(sc01) + 1)
  rowData(sc01)$feature_symbol <- rownames(sc01)
  colData(sc01)$cell_type1 <- sc01.seurat@ident
  sc01 <- selectFeatures(sc01)
  sc01 <- indexCluster(sc01)
  
  sc03.seurat <- readRDS(file.path(SCE_CACHE_DIR, 'SC03.rds'))
  sc03.clusters <- read.csv(file.path(METADATA_DIR, 'SC03_clusters.csv'), header=FALSE)
  sc03.clusters$V1 <- sub("C", "", sc03.clusters$V1)
  sc03 <- as(sc03.seurat, "SingleCellExperiment")
  counts(sc03) <- as.matrix(assay(sc03, "raw.data"))
  logcounts(sc03) <- log2(counts(sc03) + 1)
  rowData(sc03)$feature_symbol <- rownames(sc03)
  colData(sc03)$cell_type1 <- sc03.seurat@ident
  
  scmap_result <- scmapCluster(projection=sc03, 
                               index_list=list(
                                 ref=metadata(sc01)$scmap_cluster_index
                               ))
  labels1 <- sc03.clusters$V2[match(colData(sc03)$cell_type1, sc03.clusters$V1)]
  labels2 <- sc01.clusters$V2[match(scmap_result$combined_labs, sc01.clusters$V1)]
  levels(labels2) <- c(levels(labels2), "Unassigned")
  labels2[is.na(labels2)] <- "Unassigned"
  plot(getSankey(labels1, labels2, plot_width=800, plot_height=800))
}

main()
