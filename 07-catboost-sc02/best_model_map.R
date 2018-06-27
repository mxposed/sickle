require(Seurat)
require(scmap)


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

pdfPlot <- function(filename, plot, width=14, height=6) {
  pdf(file=file.path(CURRENT_DIR, filename), width=width, height=height)
  result <- plot()
  dev.off()
  return(result)
}

process <- function(exp) {
  sc03 <- readRDS(file.path(SCE_CACHE_DIR, paste(toupper(exp), 'rds', sep='.')))
  sc03.clusters <- read.csv(file.path(METADATA_DIR, paste0(exp, '_clusters.csv')), header=FALSE)
  sc03.clusters$V1 <- sub("C", "", sc03.clusters$V1)
  
  sc02.clusters <- read.csv(file.path(METADATA_DIR, 'SC02_clusters.csv'), header=FALSE)
  sc02.clusters$V1 <- sub("C", "", sc02.clusters$V1)
  sc02.clusters$V2 <- paste(sc02.clusters$V2, sc02.clusters$V1)
  
  preds <- read.csv(file.path(CURRENT_DIR, paste0(exp, '-preds.csv')))
  preds$cluster <- apply(preds[,-1], 1, which.max) - 1
  preds$X0 <- sub("-1", "", preds$X0)
  preds <- merge(sc03@cell.names, preds, by.x="x", by.y="X0")
  sc03 <- StashIdent(sc03, save.name = 'orig')
  sc03 <- SetIdent(sc03, 
                   ident.use = sc02.clusters$V2[match(
                     preds$cluster, 
                     sc02.clusters$V1
                   )]
  )
  pdfPlot(paste0(exp, '-tsne.pdf'), function() {
    TSNEPlot(sc03)
  })
  
  labels1 <- sc03.clusters$V2[match(sc03@meta.data$orig, sc03.clusters$V1)]
  #labels2 <- mca.clusters$Annotation[match(sc01@ident, mca.clusters$ClusterID)]
  #levels(labels2) <- c(levels(labels2), "Unassigned")
  #labels2[is.na(labels2)] <- "Unassigned"
  plot(getSankey(labels1, 
                 sc03@ident, 
                 plot_width = 800, 
                 plot_height = 800, 
                 colors = substr(rainbow(length(unique(labels1))), 1, 7)
  ))
}

main <- function() {
  #process('sc01')
  #process('sc02')
  process('sc03')
}

main()