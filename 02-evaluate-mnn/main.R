SCE_CACHE_DIR <- file.path(dirname(dirname(sys.frame(1)$ofile)), '01-cluster-sc01-sc02')
CACHE_DIR <- dirname(sys.frame(1)$ofile)

pdfPlot <- function(filename, plot, width=8, height=6) {
  pdf(file=file.path(CACHE_DIR, filename))
  result <- plot()
  dev.off()
  return(result)
}

main <- function() {
  cache_file <- file.path(CACHE_DIR, 'comb.rds')
  sc01 <- readRDS(file.path(SCE_CACHE_DIR, 'SC01.rds'))
  sc03 <- readRDS(file.path(SCE_CACHE_DIR, 'SC03.rds'))
  
  if (file.exists(cache_file)) {
    comb <- readRDS(cache_file)
  } else {
    sc01@project.name <- "SC01"
    sc03@project.name <- "SC03"
    sc01@meta.data$orig.ident <- "SC01"
    sc03@meta.data$orig.ident <- "SC03"
    comb <- MergeSeurat(sc01, sc03, do.normalize = FALSE, add.cell.id1 = 'SC01', add.cell.id2 = 'SC03')
    
    # computes more than 2 hrs
    corrected_file <- file.path(CACHE_DIR, 'corrected.rds')
    if (file.exists(corrected_file)) {
      corrected <- readRDS(corrected_file)
    } else {
      corrected <- mnnCorrect(
        as.matrix(comb@data[,comb@meta.data$orig.ident == "SC01"]), 
        as.matrix(comb@data[,comb@meta.data$orig.ident == "SC03"])
      )
      saveRDS(corrected, corrected_file)
    }
    
    comb@scale.data <- cbind(corrected$corrected[[1]], corrected$corrected[[2]])
    colnames(comb@scale.data) <- comb@cell.names
    comb <- RunPCA(comb, pc.genes = rownames(comb@data), do.print = FALSE, pcs.compute = 40)
    comb <- RunTSNE(comb, dims.use=1:15, do.fast=TRUE)
    saveRDS(comb, cache_file)
  }

  
  pdfPlot("comb-tsne.pdf", function() {
    TSNEPlot(comb)  
  })
  pdfPlot("comb-tsne-sc03-plasma.pdf", function() {
    TSNEPlot(comb, cells.highlight=paste("SC03", sc03@cell.names[sc03@ident == 7], sep="_"))  
  })
  pdfPlot("comb-pc-elbow.pdf", function() {
    plot(PCElbowPlot(comb, num.pc=40))
  })

}

main()