#!/bin/bash

rm -rf figs
mkdir figs


cp nested-cv.pdf figs/

cp 01-cluster-sc01-sc02/sc02v2_tsne_final.pdf figs/sc02-tsne.pdf
cp 01-cluster-sc01-sc02/sc02_cluster10.pdf figs/sc02-cluster10.pdf

cp 04-evaluate-scmap/sc03-cluster-sankey.pdf figs/sc03-to-scmap.pdf

cp 05-catboost-eva/cv3-confusion.pdf figs/

cp 07-catboost-sc02/sc01-sankey.pdf figs/sc01-to-sc02.pdf
cp 07-catboost-sc02/sc03v2-bcells-heatmap.pdf figs/sc03-bcells.pdf
cp 07-catboost-sc02/sc03v2-plasma-heatmap.pdf figs/sc03-plasma.pdf

cp 08-unseen-sc02/sc03-it50-oth4-sankey.pdf figs/sc03-to-ensemble.pdf
cp 08-unseen-sc02/sc03-it50-oth4-plasma-heatmap.pdf figs/sc03-plasma-ensemble.pdf

cp 10-catboost-sc01/sc02-sankey.pdf figs/sc02-to-sc01.pdf

cp 11-catboost-sc03/sc01-jchain.pdf figs/
cp 11-catboost-sc03/sc02-jchain.pdf figs/

cp 12-catboost-mca/sc01-sankey.pdf figs/sc01-to-mca.pdf
cp 12-catboost-mca/sc02-sankey.pdf figs/sc02-to-mca.pdf
cp 12-catboost-mca/correlation1.pdf figs/
cp 12-catboost-mca/correlation2.pdf figs/
