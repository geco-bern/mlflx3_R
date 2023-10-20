# Feedback on startup
message("Leave-Site-Out routine...")

# NOTES:
# set_opt_hparams(lr = 0.003) |>
# use lr_finder() to find optimal learning rate

# set both the R seed
# and the torch seed (both function independently)
set.seed(1)
torch::torch_manual_seed(42)

# required libraries
library(torch)
library(luz)
library(dplyr)
source("analysis/rnn_model_no_embeddings.R")
source("analysis/get_dataset_no_embeddings.R")

# automatically use the GPU if available
device <- torch::torch_device(
  if (torch::cuda_is_available()) "cuda" else "cpu"
  )

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
#lapply(sites, function(site){

site <- sites

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
      train_center
    )

  # format torch data loader
  # for test data
  test_ds <- df |>
    dplyr::filter(
      sitename == !!site
    ) |>
    gpp_dataset(
      train_center
    )

  # run data loaders, batch
  # size is limited to 1 as
  # the dimensions of the input
  # should be equal to take advantage
  # of batch processing (probably due
  # to underlying matrix optimizations)
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

  # save checkpoints on epochs of train loss
  # checkpoint <- luz::luz_callback_model_checkpoint(
  #    path = file.path(here::here("data/checkpoints/", site)),
  #    monitor = "train_loss"
  # )

  # fit the model
  fitted <- rnn_model |>
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam,
    metrics = list(luz_metric_mae())
  ) |>
    set_hparams(
      input_size = 11,
      hidden_size = 256,
      output_size = 1
    ) |>
  fit(
    train_dl,
    epochs = 100,
    valid_data = test_dl
    #callbacks = list(checkpoint)
    )

  # save model for this iteration
  # i.e. site left out
  luz_save(
    fitted,
    file.path(
      here::here("data/leave_site_out/",
                 paste0(site, ".pt"))
      )
    )

#})
