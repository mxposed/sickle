require(Seurat)


thisFile <- function() {
  cmdArgs <- commandArgs(trailingOnly = FALSE)
  needle <- "--file="
  match <- grep(needle, cmdArgs)
  if (length(match) > 0) {
    # Rscript
    return(normalizePath(sub(needle, "", cmdArgs[match])))
  } else {
    # 'source'd via R console
    return(normalizePath(sys.frames()[[1]]$ofile))
  }
}

CURRENT_DIR <- dirname(thisFile())
CODE_ROOT <- dirname(CURRENT_DIR)
SCE_CACHE_DIR <- file.path(CODE_ROOT, '01-cluster-sc01-sc02')
METADATA_DIR <- file.path(CODE_ROOT, '00-metadata')

pdfPlot <- function(filename, plot, width=8, height=6) {
  pdf(file=file.path(CURRENT_DIR, filename), width=width, height=height)
  result <- plot()
  dev.off()
  return(result)
}

cluster <- function(dataset, save_as=NULL, num_pcs=NULL, resolution=0.5) {
  if (is.null(save_as)) {
    save_as <- dataset
  }
  cache_file <- paste(CURRENT_DIR, paste(save_as, "rds", sep='.'), sep='/')
  
  if (file.exists(cache_file)) {
    sce <- readRDS(cache_file)
  } else {
    data <- Read10X(data.dir = file.path(CODE_ROOT, "..", dataset, sep='/'))
    sce <- CreateSeuratObject(raw.data=data, min.cells=3, min.genes=200)
    mito.genes <- grep(pattern="^mt-", x=rownames(sce@data), value=TRUE)
    percent.mito <- Matrix::colSums(sce@raw.data[mito.genes, ])/Matrix::colSums(sce@raw.data)
    sce <- AddMetaData(sce, metadata=percent.mito, col.name="percent.mito")
    sce <- FilterCells(sce, subset.names = c("nGene", "percent.mito"), 
                       low.thresholds = c(300, -Inf), 
                       high.thresholds = c(4000, 0.1))
    sce <- NormalizeData(sce)
    sce <- ScaleData(sce, vars.to.regress = c("nUMI", "percent.mito"))
    sce <- FindVariableGenes(sce, x.low.cutoff=0.0125, x.high.cutoff=3, y.cutoff=0.5, do.plot = FALSE)
    sce <- RunPCA(sce, pc.genes = sce@var.genes, do.print = FALSE, pcs.compute = 40)
    sce <- ProjectPCA(sce, do.print=FALSE)
  }
  
  pdfPlot(paste0(save_as, '_elbow.pdf'), function() {
    plot(PCElbowPlot(sce, num.pc = 40))
  })
  
  sce <- RunTSNE(sce, dims.use = 1:num_pcs, check_duplicates = FALSE, do.fast=TRUE)
  sce <- FindClusters(sce, dims.use = 1:num_pcs, resolution=resolution, 
                      print.output = FALSE, save.SNN = TRUE, force.recalc = TRUE)
  
  #sce <- SetIdent(sce, cells.use = WhichCells(sce, ident=3), 0)
  sce <- ValidateSpecificClusters(sce, 2, 0)
  ident <- as.numeric(levels(sce@ident)[sce@ident])
  ident[ident > 2] <- ident[ident > 2] - 1
  sce <- SetIdent(sce, ident.use = ident)
  
  write.csv(sce@ident, file.path(CURRENT_DIR, paste0(save_as, '_assgn.csv')))
  
  pdfPlot(paste0(save_as, '_tsne.pdf'), function() {
    TSNEPlot(sce, do.label = TRUE)
  }, width=12, height=10)
  
  saveRDS(sce, cache_file)
  return(sce)
}

markers <- function(sce, save_as, num_pcs, resolution) {
  markers <- FindAllMarkers(sce, only.pos = TRUE, 
                            min.pct = 0.25, thresh.use = 0.5, 
                            max.cells.per.ident = 200)
  markers_file <- file.path(CURRENT_DIR, 
                            paste0(save_as, "_markers_PC1-", num_pcs, "res", resolution, ".csv"))
  write.csv(markers, markers_file)
}

main <- function() {
  sc02 <- cluster("SC02", save_as="SC02v2", num_pcs = 25, resolution = 0.5)
  #markers(sc02, "SC02v2", num_pcs = 25, resolution = 0.5)
}

main()
