# feedback
message("Leave-Site-Out routine...")
set.seed(42)

# required libraries
library(torch)
library(luz)
library(dplyr)
source("analysis/rnn_model.R")
source("analysis/get_dataset.R")

# automatically use the GPU if available
device <- torch::torch_device(
  if (torch::cuda_is_available()) "cuda" else "cpu"
  )

# features / epochs / ...
num_epochs <- 150
input_features <- 11
hidden_dim <- 256
conditional_features <- 21 # static across time

# read in data, only retain relevant features
df <- readRDS("data/df_imputed.rds") |>
  dplyr::select(
    'sitename',
    'GPP_NT_VUT_REF',
    'TA_F',
    'SW_IN_F',
    'TA_F_DAY',
    'LW_IN_F',
    'WS_F',
    'P_F',
    'VPD_F',
    'TA_F_NIGHT',
    'PA_F',
    'wscal',
    'fpar'
  )

# leave site out routine
sites <- unique(df$sitename)[1]

# loop over all sites (drop the one mentioned)
# use a lapply rather than a for loop as lapply
# does not create global variables which might
# get recycled if not carefully purged
lapply(sites, function(site){

  # Initiate model and other parts
  net <- rnn_model(
    input_size = input_features,
    hidden_size = hidden_dim,
    conditional = 0,
    conditional_size = conditional_features,
    output_size = 1
  )

  # set device
  net <- net$to(device = device)

  # set optimizer
  optimizer <- optim_adam(net$parameters, lr = 0.001)

  # define train and validation routines
  # use 'luz' in the future
  train_batch <- function(b) {

    optimizer$zero_grad()
    output <- net(b$x$to(device = device))
    target <- b$y$to(device = device) |>
      torch_squeeze()

    loss <- nnf_mse_loss(output, target)
    loss$backward()
    optimizer$step()
    loss$item()
  }

  valid_batch <- function(b) {
    output <- net(b$x$to(device = device))

    # check the squeeze to make dimensions match
    target <- b$y$to(device = device)|>
      torch_squeeze()

    loss <- nnf_mse_loss(output, target)
    loss$item()
  }

  # split out leave one site out
  # training and testing data
  train <- df |>
    dplyr::filter(
      sitename != !!site
    )

  # calculated mean / sd to center
  # the data
  train_center <- train |>
    summarise(
      across(
        where(is.numeric),
        list(mean = mean, sd = sd)
        )
      ) |>
    ungroup()

  # format torch data loader
  # for training data
  train_ds <- train |>
    gpp_dataset(
      train_center,
      1
    )

  # format torch data loader
  # for test data
  test_ds <- df |>
    dplyr::filter(
      sitename == !!site
    ) |>
    gpp_dataset(train_center, 1)

  # run data loaders
  train_dl <- dataloader(
    train_ds,
    batch_size = 1,
    shuffle = TRUE
  )

  test_dl <- dataloader(
    test_ds,
    batch_size = 1,
    shuffle = TRUE
  )

  for (epoch in 1:num_epochs) {

    net$train()
    train_loss <- c()

    coro::loop(for (b in train_dl) {
      loss <-train_batch(b)
      train_loss <- c(train_loss, loss)
    })

    cat(sprintf("\nEpoch %d, training: loss: %3.5f \n", epoch, mean(train_loss)))

    net$eval()
    valid_loss <- c()

    coro::loop(for (b in test_dl) {
      loss <- valid_batch(b)
      valid_loss <- c(valid_loss, loss)
    })

    cat(sprintf("\nEpoch %d, validation: loss: %3.5f \n", epoch, mean(valid_loss)))
  }
})
