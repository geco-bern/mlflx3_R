# Torch ML

Torch ML time series prediction tests, trying to get to some sort of reproducibility.

This will be a full port to R, abandoning python and probably optimizing the workflow and ML model for speed while at it.

## Docker images

For reproducible results and portability rocker images should be considered.

https://rocker-project.org/

A custom docker is provided which includes the GPU torch setup

```
docker build -f custom_ml.Dockerfile -t rocker-torch .
docker run --gpus all -p 8787:8787 rocker-torch
```

```
# CPU-only
docker run -p 8787:8787 rocker/ml
# Machines with nvidia-docker and GPU support
docker run --gpus all -p 8787:8787 rocker/ml
```

## Torch install

By default the CUDA install is ignored.

Install torch using CRAN release

```
install.packages(torch)
```

Then force the version to 11.7 (as 11.8 is not yet available).

```
Sys.setenv(CUDA="11.7")
torch::install_torch()
```


## Mount external volume for running the project

```
docker run --gpus all -e PASSWORD="rstudio" -p 8787:8787 -v $(pwd)/work:/home/rstudio/ml_project -e ROOT=TRUE rocker-torch
```

