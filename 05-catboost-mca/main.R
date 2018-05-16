require(Seurat)
require(scmap)

CURRENT_DIR <- dirname(sys.frame(1)$ofile)
CODE_ROOT <- dirname(CURRENT_DIR)
SCE_CACHE_DIR <- file.path(CODE_ROOT, '01-cluster-sc01-sc02')
METADATA_DIR <- file.path(CODE_ROOT, '00-metadata')

pdfPlot <- function(filename, plot, width=8, height=6) {
  pdf(file=file.path(CURRENT_DIR, filename))
  result <- plot()
  dev.off()
  return(result)
}

main <- function() {
  sc01 <- readRDS(file.path(SCE_CACHE_DIR, 'SC01.rds'))
  sc01.clusters <- read.csv(file.path(METADATA_DIR, 'SC01_clusters.csv'), header=FALSE)
  sc01.clusters$V1 <- sub("C", "", sc01.clusters$V1)
  
  mca.clusters <- read.csv(file.path(CURRENT_DIR, 'MCA_clusters.csv'))
  
  preds <- read.csv(file.path(CURRENT_DIR, 'sc01-preds.csv'))
  preds$X0 <- sub("-1", "", preds$X0)
  preds <- merge(sc01@cell.names, preds, by.x="x", by.y="X0")
  sc01 <- StashIdent(sc01, save.name = 'orig')
  sc01 <- SetIdent(sc01, ident.use = preds$X0.1)
  pdfPlot('sc01-catboost-tsne.pdf', function() {
    TSNEPlot(sc01)
  })
  pdfPlot('sc01-catboost-probs.pdf', function() {
    hist(preds$prob, breaks=100)
  })
  
  labels1 <- sc01.clusters$V2[match(sc01@meta.data$orig, sc01.clusters$V1)]
  labels2 <- mca.clusters$Annotation[match(sc01@ident, mca.clusters$ClusterID)]
  levels(labels2) <- c(levels(labels2), "Unassigned")
  labels2[is.na(labels2)] <- "Unassigned"
  plot(getSankey(labels1, labels2, plot_width = 800, plot_height = 800))
}

main()