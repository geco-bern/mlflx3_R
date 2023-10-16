# Torch ML

Torch ML time series prediction tests, trying to get to some sort of reproducibility.

This will be a full port to R, abandoning python and probably optimizing the workflow and ML model for speed while at it.

## Docker images

For reproducible results and portability rocker images should be considered.

https://rocker-project.org/

```
# CPU-only
docker run -p 8787:8787 rocker/ml
# Machines with nvidia-docker and GPU support
docker run --gpus all -p 8787:8787 rocker/ml
```
