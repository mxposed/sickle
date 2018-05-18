require(Seurat)
require(scmap)

CURRENT_DIR <- dirname(sys.frame(1)$ofile)
CODE_ROOT <- dirname(CURRENT_DIR)
SCE_CACHE_DIR <- file.path(CODE_ROOT, '01-cluster-sc01-sc02')
METADATA_DIR <- file.path(CODE_ROOT, '00-metadata')

pdfPlot <- function(filename, plot, width=14, height=6) {
  pdf(file=file.path(CURRENT_DIR, filename), width=width, height=height)
  result <- plot()
  dev.off()
  return(result)
}

process <- function(exp) {
  sc01 <- readRDS(file.path(SCE_CACHE_DIR, paste(toupper(exp), 'rds', sep='.')))
  sc01.clusters <- read.csv(file.path(METADATA_DIR, paste0(exp, '_clusters.csv')), header=FALSE)
  sc01.clusters$V1 <- sub("C", "", sc01.clusters$V1)
  
  mca.clusters <- read.csv(file.path(CURRENT_DIR, 'MCA_clusters.csv'))
  
  preds <- read.csv(file.path(CURRENT_DIR, paste0(exp, '-best-preds.csv')))
  preds$cluster <- apply(preds[,-1],1,which.max)
  preds$X0 <- sub("-1", "", preds$X0)
  preds <- merge(sc01@cell.names, preds, by.x="x", by.y="X0")
  sc01 <- StashIdent(sc01, save.name = 'orig')
  sc01 <- SetIdent(sc01, 
                   ident.use = mca.clusters$Annotation[match(
                     preds$cluster, 
                     mca.clusters$ClusterID
                   )]
                  )
  pdfPlot(paste0(exp, '-best-tsne.pdf'), function() {
    TSNEPlot(sc01)
  })
  
  labels1 <- sc01.clusters$V2[match(sc01@meta.data$orig, sc01.clusters$V1)]
  #labels2 <- mca.clusters$Annotation[match(sc01@ident, mca.clusters$ClusterID)]
  #levels(labels2) <- c(levels(labels2), "Unassigned")
  #labels2[is.na(labels2)] <- "Unassigned"
  plot(getSankey(labels1, sc01@ident, plot_width = 800, plot_height = 800))
}

main <- function() {
  process('sc01')
  process('sc02')
  process('sc03')
}

main()