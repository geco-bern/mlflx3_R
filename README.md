# Modelling flux responses to drought using DNN

This analysis explores if LSTM models have any advantage over conventional neural nets when modelling GPP. Given that fAPAR is included in the predictors any conventional wisdom suggests that lagged dependencies are encoded in this variable. Splitting out some of this variability can only be accomplished in cases where drought would limit the productivity without altering the state of the canopy, i.e. evergreen trees. It is therefore assumed hypothesized that the LSTM architecture might perform better for evergreen trees which are exposed to drought conditions.

The workflow will use a leave-site-out cross validation as a robustness check, i.e. training the model on all but the left out site. The left out site will be used as a test case. Data are split between drought decidious and evegreen droughted locations as previously described in Stocker et al. 2018 (see figure 6).

## Setup

Note that this refactored code only reproduces a part of the complete workflow. However, code shown should allow for the quick adaptation of it to run different stratification (cross-validation) scenarios. The code shown here runs the most expensive leave-site-out cross validation, which makes up the bulk of the analysis in the manuscript.

It is adviced to run this code on an accelerated setup (CUDA GPU). To ensure consistency across platforms, and not deal with a zoo of required CUDA drivers which can conflict due to platform and ML platforms already in use I suggest to use the included docker file (and environment). To install and use docker on your system I refer to the [docker documentation](https://www.docker.com/).

### Docker images

The dockerfile included provides a GPU torch setup. You can build
this docker image using the below command. This will download the NVIDIA CUDA
drivers for GPU support, the tidyverse, rstudio IDE and quarto publishing
environment. Note that this setup will require some time to build given the
the large downloads involved. Once build locally no further downloads will be
required.

```
# In the main project directory run
docker build -f Dockerfile -t rocker-torch .
```

To spin up a GPU docker image use in the project directory:

```
docker run --gpus all -e PASSWORD="rstudio" -p 5656:8787 -v $(pwd):/workspace rocker-torch
```

In any browser use the [http://localhost:5656](http://localhost:5656) 
url to access the docker RStudio Server instance which should be running. The password to the RStudio Server instance is set to `rstudio` when using the
above commands (but can and should be changed if the computer is exposed to a
larger institutional network). This is not a secured setup, use a stronger password or a local firewall to avoid abuse.

Data will be mounted in the docker virtual machine at `/workspace` and is fully accessible (writing and reading of files on your local file system).

### Running the analysis

Analysis scripts are stored in the `analysis` folder and should be run in sequence. Note that the standard deep neural net is small enough to run an CPU. Summary [results](https://geco-bern.github.io/mlflx3_R/articles/01_results.html) are automatically rendered using the vignettes in the `vignettes` folder.

## References

Stocker, Benjamin D., Jakob Zscheischler, Trevor F. Keenan, I. Colin Prentice, Josep Peñuelas, and Sonia I. Seneviratne. “Quantifying Soil Moisture Impacts on Light Use Efficiency across Biomes.” New Phytologist 218, no. 4 (June 2018): 1430–49. https://doi.org/10.1111/nph.15123.

