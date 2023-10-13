# feedback
message("Leave-Site-Out routine...")
set.seed(42)

# required libraries
library(torch)
library(dplyr)
source("analysis/rnn_model.R")

# features / epochs / ...
epochs <- 150
input_features <- 11
hidden_dim <- 256
conditional_features <- 21 # static across time
train_loss <- 0.0
train_r2 <- 0.0

# read in data
df <- readRDS("data/df_imputed.rds")

# # initiate the model setup outside the
# # loop (only data will be changed)
# net <- rnn_model(
#   input_size = 1,
#   hidden_size = 32,
#   linear_size = 512,
#   output_size = n_forecast
# )
#
# # automatically use the GPU if available
# device <- torch::torch_device(if (torch::cuda_is_available()) "cuda" else "cpu")
# net <- net$to(device = device)
#
# # set optimizer
# optimizer <- optim_adam(net$parameters, lr = 0.001)

# train_batch <- function(b) {
#
#   optimizer$zero_grad()
#   output <- net(b$x$to(device = device))
#   target <- b$y$to(device = device)
#
#   loss <- nnf_mse_loss(output, target)
#   loss$backward()
#   optimizer$step()
#
#   loss$item()
# }
#
# valid_batch <- function(b) {
#
#   output <- net(b$x$to(device = device))
#   target <- b$y$to(device = device)
#
#   loss <- nnf_mse_loss(output, target)
#   loss$item()
#
# }

# write config to file with basic optimization and model
# settings

# leave site out routine
sites <- unique(df$sitename)

# loop over all sites (drop the one mentioned)
# use a lapply rather than a for loop as lapply
# does not create global variables which might
# get recycled if not carefully purged
lapply(sites, function(site){

  # split out leave one site out
  # training and testing data
  train <- df |>
    dplyr::filter(
      sitename != !!site
    )

  test <- df |>
    dplyr::filter(
      sitename == !!site
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

    coro::loop(for (b in valid_dl) {
      loss <- valid_batch(b)
      valid_loss <- c(valid_loss, loss)
    })

    cat(sprintf("\nEpoch %d, validation: loss: %3.5f \n", epoch, mean(valid_loss)))
  }

})
