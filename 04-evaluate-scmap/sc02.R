require(scmap)
require(SingleCellExperiment)
require(SeuratConverter)
require(irr)

CURRENT_DIR <- dirname(sys.frame(1)$ofile)
CODE_ROOT <- dirname(CURRENT_DIR)
SCE_CACHE_DIR <- file.path(CODE_ROOT, '01-cluster-sc01-sc02')
METADATA_DIR <- file.path(CODE_ROOT, '00-metadata')


getSC02 <- function() {
  cache <- file.path(CURRENT_DIR, 'sc02.rds')
  if (!file.exists(cache)) {
    sc02.seurat <- readRDS(file.path(SCE_CACHE_DIR, 'SC02v2.rds'))
    sc02 <- as(sc02.seurat, "SingleCellExperiment")
    counts(sc02) <- as.matrix(assay(sc02, "raw.data"))
    logcounts(sc02) <- log2(counts(sc02) + 1)
    rowData(sc02)$feature_symbol <- rownames(sc02)
    colData(sc02)$cell_type1 <- sc02.seurat@ident
    sc02 <- selectFeatures(sc02)
    sc02 <- indexCluster(sc02)
    sc02 <- indexCell(sc02)
    saveRDS(sc02, cache)
  } else {
    sc02 <- readRDS(cache)
  }
  return(sc02)
}


getSC03 <- function() {
  cache <- file.path(CURRENT_DIR, 'sc03.rds')
  if (!file.exists(cache)) {
    sc03.seurat <- readRDS(file.path(SCE_CACHE_DIR, 'SC03.rds'))
    sc03 <- as(sc03.seurat, "SingleCellExperiment")
    counts(sc03) <- as.matrix(assay(sc03, "raw.data"))
    logcounts(sc03) <- log2(counts(sc03) + 1)
    rowData(sc03)$feature_symbol <- rownames(sc03)
    colData(sc03)$cell_type1 <- sc03.seurat@ident
    saveRDS(sc03, cache)
  } else {
    sc03 <- readRDS(cache)
  }
  return(sc03)
}


main <- function() {
  sc02.clusters <- read.csv(file.path(METADATA_DIR, 'SC02v2_clusters.csv'), header=FALSE)
  sc02.clusters$V1 <- sub("C", "", sc02.clusters$V1)
  
  sc03.clusters <- read.csv(file.path(METADATA_DIR, 'SC03_clusters.csv'), header=FALSE)
  sc03.clusters$V1 <- sub("C", "", sc03.clusters$V1)
  
  sc02 <- getSC02()
  sc03 <- getSC03()
  
  scmap_result <- scmapCluster(projection=sc03, 
                               index_list=list(
                                 ref=metadata(sc02)$scmap_cluster_index
                               ))
  labels1 <- sc03.clusters$V2[match(colData(sc03)$cell_type1, sc03.clusters$V1)]
  labels2 <- sc02.clusters$V2[match(scmap_result$combined_labs, sc02.clusters$V1)]
  names(labels2) <- rownames(colData(sc03))
  levels(labels2) <- c(levels(labels2), "Novel cell type")
  labels2[is.na(labels2)] <- "Novel cell type"
  write.csv(labels2, file.path(CURRENT_DIR, 'sc03-cluster-preds.csv'))
  
  scmap_result <- scmapCell(projection=sc03, 
                            index_list=list(
                              ref=metadata(sc02)$scmap_cell_index
                            ))
  scmap_clusters <- scmapCell2Cluster(
    scmap_result, 
    list(as.character(colData(sc02)$cell_type1))
  )
  labels2 <- sc02.clusters$V2[match(scmap_clusters$combined_labs, sc02.clusters$V1)]
  names(labels2) <- rownames(colData(sc03))
  levels(labels2) <- c(levels(labels2), "Novel cell type")
  labels2[is.na(labels2)] <- "Novel cell type"
  write.csv(labels2, file.path(CURRENT_DIR, 'sc03-cell-preds.csv'))
}

main()
