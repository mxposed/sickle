require(Seurat)

DATASET_DIR <- "/Users/markov/Documents/MSc Bioinformatics/scRNA clustering/dataset 1/"
CURRENT_DIR <- dirname(sys.frame(1)$ofile)

pdfPlot <- function(filename, plot, width=8, height=6) {
  pdf(file=file.path(CURRENT_DIR, filename), width=width, height=height)
  result <- plot()
  dev.off()
  return(result)
}

cluster <- function(dataset, num_pcs=NULL, resolution=0.5) {
  cache_file <- paste(CURRENT_DIR, paste(dataset, "rds", sep='.'), sep='/')
  
  if (file.exists(cache_file)) {
    sce <- readRDS(cache_file)
  } else {
    data <- Read10X(data.dir = paste(DATASET_DIR, dataset, sep='/'))
    sce <- CreateSeuratObject(raw.data=data, min.cells=3, min.genes=200)
    mito.genes <- grep(pattern="^mt-", x=rownames(sce@data), value=TRUE)
    percent.mito <- Matrix::colSums(sce@raw.data[mito.genes, ])/Matrix::colSums(sce@raw.data)
    sce <- AddMetaData(sce, metadata=percent.mito, col.name="percent.mito")
    sce <- FilterCells(sce, subset.names = c("nGene", "percent.mito"), 
                       low.thresholds = c(300, -Inf), 
                       high.thresholds = c(4000, 0.1))
    sce <- NormalizeData(sce)
    sce <- ScaleData(sce, vars.to.regress = c("nUMI", "percent.mito"))
    sce <- FindVariableGenes(sce, x.low.cutoff=0.0125, x.high.cutoff=3, y.cutoff=0.5)
    sce <- RunPCA(sce, pc.genes = sce@var.genes, do.print = TRUE, pcs.print = 1:5, 
                  genes.print = 5, pcs.compute = 40)
    sce <- ProjectPCA(sce, do.print=FALSE)
  }
  
  pdfPlot(paste0(dataset, '_elbow.pdf'), function() {
    plot(PCElbowPlot(sce, num.pc=40))
  })
  
  if (is.null(num_pcs)) {
    saveRDS(sce, cache_file)
    return(sce)
  }
  
  sce <- FindClusters(sce, dims.use = 1:num_pcs, resolution=resolution, 
                      print.output = FALSE, save.SNN = TRUE)
  sce <- RunTSNE(sce, dims.use = 1:num_pcs, check_duplicates = FALSE)
  
  write.csv(sce@ident, file.path(CURRENT_DIR, paste0(dataset, '_assgn.csv')))
  
  pdfPlot(paste0(dataset, '_tsne.pdf'), function() {
    TSNEPlot(sce, do.label = TRUE)
  })
  
  markers <- FindAllMarkers(sce, only.pos = TRUE, 
                            min.pct = 0.25, thresh.use = 0.5, 
                            max.cells.per.ident = 200)
  markers_file <- paste(CURRENT_DIR, 
                        paste0(dataset, "_markers_PC1-", num_pcs, "res", resolution, ".csv"),
                        sep='/')
  write.csv(markers, markers_file)
  
  saveRDS(sce, cache_file)
  return(sce)
}

main <- function() {
  #cluster("SC01", num_pcs = 28)
  cluster("SC02", num_pcs = 25)
  #cluster("SC03", num_pcs = 21)
}