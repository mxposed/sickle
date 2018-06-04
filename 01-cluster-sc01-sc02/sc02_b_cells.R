require(ggplot2)
require(gridExtra)
require(grid)


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

sce <- readRDS(file.path(CURRENT_DIR, 'SC02.rds'))
vars <- c('nGene', 'nUMI', 'percent.mito')
b.cells <- sce@ident %in% c(0, 3)
b.cells.df <- data.frame(sce@meta.data[b.cells, vars], ident=sce@ident[b.cells])

plots <- list()
for (f in fields) {
  plots[[f]] <- ggplot(b.cells.df, aes_string(f, fill="ident")) +
                geom_histogram(alpha=0.5, aes(y=..density..), position='identity', bins=50) +
                xlab(f)
}

ml <- arrangeGrob(grobs=plots, 
                  nrow=3, ncol=1, 
                  top=textGrob("Density of technical variables between two B-Cell clusters", 
                                gp=gpar(fontsize=16)))
ggsave(file.path(CURRENT_DIR, "SC02-b-cells-tech-vars.png"), ml, width=6, height=6)


de.genes <- FindMarkers(sce, 0, 3)
write.csv(de.genes, file=file.path(CURRENT_DIR, "SC02-b-cells-DE-genes.csv"))