require(Seurat)
require(scmap)

SCE_CACHE_DIR <- file.path(dirname(dirname(sys.frame(1)$ofile)), '01-cluster-sc01-sc02')
CACHE_DIR <- dirname(sys.frame(1)$ofile)

pdfPlot <- function(filename, plot, width=8, height=6) {
  pdf(file=file.path(CACHE_DIR, filename))
  result <- plot()
  dev.off()
  return(result)
}

main <- function() {
  sc01 <- readRDS(file.path(SCE_CACHE_DIR, 'SC01.rds'))
  preds <- read.csv(file.path(CACHE_DIR, 'sc01-preds.csv'))
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
  plot(getSankey(sc01@meta.data$orig, sc01@ident))
}

main()