require(Seurat)
require(ggplot2)


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

zz <- function(sc02) {
  p <- DoHeatmap(
    sc02, 
    genes.use = c("Irf8", "Cd209a", "Itgae", "Ccr7"), 
    cells.use = WhichCells(sc02, 10),
    slim.col.label = TRUE,
    group.by = NULL,
    draw.line = FALSE,
    group.spacing = 0
  )
  data <- sc02@scale.data[c("Irf8", "Cd209a", "Itgae", "Ccr7"), WhichCells(sc02, 10)]
  data_gene <- 10 - apply(data, 2, which.max)
  data_max <- apply(data, 2, max)
  custom_order <- names(data_max)[order(data_gene, data_max, decreasing = TRUE)]
  p$data$cell <- factor(p$data$cell, levels = custom_order)
  p <- p + labs(x="Cells from cluster #10 in SC02", tag="B") +
    theme(
      axis.title.x = element_text(size=14), 
      axis.text.y = element_text(size=14),
      plot.tag.position = "topright",
      plot.tag = element_text(size = 20)
    ) + guides(fill=guide_colorbar(
      barheight=7.8, 
      title = "Gene expression z-score", 
      title.position = "right", 
      title.theme = element_text(angle=90, size = 10),
      label.theme = element_text(size=8, angle = 0),
      title.vjust = 0.5
    )) + 
    geom_tile(colour=NA)
  plot(p)
}

qq <- function(sc02) {
  p <- TSNEPlot(
    sc02, 
    do.label = TRUE, 
    do.return = TRUE, 
    pt.size = 0.1,
    label.size = 3.5
  )
  p$layers[[1]]$aes_params$alpha = 0.7
  p <- p + theme(
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    axis.title = element_text(size = 8),
    plot.tag.position = "topright",
    plot.tag = element_text(size = 20)
  ) + guides(colour=guide_legend(
    ncol = 2,
    override.aes = list(size = 3, alpha = 1),
    title = "Cluster",
    title.position = "top",
    title.theme = element_text(size = 10),
    title.hjust = 0.5
  )) + labs(tag = "A")
  plot(p)
}

main <- function() {
  sc02 <- readRDS(file.path(SCE_CACHE_DIR, 'SC02v2.rds'))

  pdfPlot('sc02_cluster10.pdf', function() {
    zz(sc02)
  }, width = 5, height = 2)
  
  pdfPlot('sc02v2_tsne_final.pdf', function() {
    qq(sc02)
  }, width = 5, height = 4)
}

main()