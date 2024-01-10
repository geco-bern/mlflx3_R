# Modelling flux responses to drough using DNN

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

Analysis scripts are stored in the `analysis` folder and should be run in sequence. Note that the standard deep neural net is small enough to run an CPU. Summary [results](https://geco-bern.github.io/mlflx2_R/articles/01_results.html) are automatically rendered using the vignettes in the `vignettes` folder.
