# An effective machine learning approach for predicting ecosystem CO2 assimilation across space and time

## Abstract:

Accurate predictions of environmental controls on ecosystem photosynthesis are essential for understanding the impacts of climate change and extreme events on the carbon cycle and the provisioning of ecosystem services. Using time-series measurements of ecosystem fluxes paired with measurements of meteorological variables from a network of globally distributed sites and remotely sensed vegetation indices, we train a recurrent deep neural network (Long-Short-Term Memory, LSTM), a simple deep neural network (DNN), and a mechanistic, theory-based photosynthesis model with the aim to predict ecosystem gross primary production (GPP). We test these models' ability to spatially and temporally generalise across a wide range of environmental conditions. Both neural network models outperform the theory-based model considering leave-site-out cross-validation (LSOCV). The LSTM model performs best and achieves a mean R2 of 0.78 across sites in the LSOCV and an average R2 of 0.82 across relatively moist temperate and boreal sites. This suggests that recurrent deep neural networks provide a basis for robust data-driven ecosystem photosynthesis modelling in respective biomes. However, limits to global model upscaling are identified using cross-validation by vegetation types and by continents. In particular, our model performance is weakest at relatively arid sites where unknown vegetation exposure to water limitation limits model reliability.

## Setup

## Docker images

A custom docker is provided which includes the GPU torch setup. You can build
this docker image using the below command. This will download the NVIDIA CUDA
drivers for GPU support, the tidyverse, rstudio IDE and quarto publishing
environment. Note that this setup will require some time to build given the
the large downloads involved. Once build locally no further downloads will be
required.

```
# In the main project directory
docker build -f Dockerfile -t rocker-torch .
```

To spin up the docker image use:

```
# CPU-only
docker run -e PASSWORD="rstudio" -p 5656:5656 -v $(pwd):/workspace -p 8787:8787 rocker/ml

# GPU support (NVIDIA)
docker run --gpus all -e PASSWORD="rstudio" -p 5656:5656 -v $(pwd):/workspace rocker-torch
```

When in the main R project folder run either the CPU or GPU command to start
the docker instance. In any browser use the [http://localhost:8787](http://localhost:8787) 
url to access the docker RStudio Server instance which should be running.

The password to the RStudio Server instance is set to `rstudio` when using the
above commands (but can and should be changed if the computer is exposed to a
larger institutional network). Anyone with the default password will have
access to your virtual machine and mounted data. Data will be mounted in the 
docker virtual machine at `/workspace` and is fully accessible (writing / 
reading of files on your local file system).

## Running the analysis

Analysis are stored in the `analysis` folder and should be run in sequence.
