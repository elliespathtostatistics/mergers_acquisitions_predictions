install.packages("edgar", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(edgar)
df <- read_csv("cik_ticker.csv")
ciks <- df$CIK
# Script was split across 8 different sbatch jobs in 3 year increments, took ~12 hours per job
years <- seq(1994,2018,1)
for (year in years) {
  for (cik in ciks) {
    try(getMgmtDisc(cik.no = c(cik), filing.year = year))
  }
}
# This downloaded 10-K reports into a file in my scratch directory, the dataset was then
# loaded onto hdfs with the command: hdfs dfs -put /scratch/rpm295/corpus bdad_final_project/
