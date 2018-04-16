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
    g.1 <- head(rownames(sc01@hvg.info), 1000)
    g.2 <- head(rownames(sc03@hvg.info), 1000)
    genes.use <- unique(c(g.1,g.2))
    genes.use <- intersect(genes.use, rownames(sc01@scale.data))
    genes.use <- intersect(genes.use, rownames(sc03@scale.data))
  
    comb <- RunCCA(sc01, sc03, genes.use = genes.use, num.cc=30,
                   add.cell.id1='SC01',
                   add.cell.id2='SC03')
    comb <- CalcVarExpRatio(comb, reduction.type="pca",
                            grouping.var="orig.ident",
                            dims.use = 1:26)
    comb <- AlignSubspace(comb, reduction.type="cca",
                          grouping.var="orig.ident",
                          dims.align = 1:26)
    comb <- RunTSNE(comb, reduction.use = "cca.aligned", dims.use=1:26, do.fast=TRUE)
  }
  pdfPlot("comb-metagene-bicor.pdf", function() {
    cache_file <- file.path(CACHE_DIR, 'bicor.rds')
    if (file.exists(cache_file)) {
      bicor.data <- readRDS(cache_file)
    } else {
      bicor.data <- MetageneBicorPlot(comb, 
                                      grouping.var = 'orig.ident', 
                                      dims.eval=1:30,
                                      return.mat=TRUE)
      saveRDS(bicor.data, file=cache_file)
    }
    MetageneBicorPlot(comb, bicor.data, grouping.var = 'orig.ident', dims.eval=1:30)
  })
  
  pdfPlot("comb-tsne.pdf", function() {
    TSNEPlot(comb)  
  })
  pdfPlot("comb-tsne-outliers.pdf", function() {
    TSNEPlot(comb, cells.highlight=comb@cell.names[comb@meta.data$var.ratio.pca < 0.8])  
  })
  pdfPlot("comb-tsne-sc03-plasma.pdf", function() {
    TSNEPlot(comb, cells.highlight=paste("SC03", sc03@cell.names[sc03@ident == 7], sep="_"))  
  })
  pdfPlot("comb-hist-outliers.pdf", function() {
    hist(comb@meta.data$var.ratio.pca[comb@meta.data$var.ratio.pca < 2], breaks=50)  
  })
}

main()