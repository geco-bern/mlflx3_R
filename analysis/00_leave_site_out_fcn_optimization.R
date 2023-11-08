# Feedback on startup
message("Leave-Site-Out routine...")

# set both the R seed
# and the torch seed (both function independently)
set.seed(1)
torch::torch_manual_seed(42)
epochs <- 150 # for testing purposes, set to 150 for full run

# required libraries
library(torch)
library(luz)
library(dplyr)
source("R/fcn_model.R")
source("R/gpp_dataset.R")

# automatically use the GPU if available
device <- torch::torch_device(
  if (torch::cuda_is_available()) "cuda" else "cpu"
)

# read in data, only retain relevant features
df <- readRDS("data/df_imputed.rds") |>
  dplyr::select(
    'sitename',
    'date',
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
sites <- unique(df$sitename)

# loop over all sites (drop the one mentioned)
# use a lapply rather than a for loop as lapply
# does not create global variables which might
# get recycled if not carefully purged
leave_site_out_output <- lapply(sites, function(site){

  # check if run was already finished
  # if sites are skipped run the offline
  # routine to colate all data
  if(file.exists(
    here::here("data/leave_site_out_weights_fcn/",
               paste0(site, ".pt"))
  )) {
    message(sprintf("run completed, skipping %s ...", site))
    return(NULL)
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
  # to underlying matrix optimization)
  train_dl <- dataloader(
    train_ds,
    batch_size = 1,
    shuffle = TRUE
  )

  test_dl <- dataloader(
    test_ds,
    batch_size = 1,
    shuffle = FALSE
  )

  # fit the model by defining
  # a setup, setting parameters
  # and then initiating the fitting
  fitted <- fcn_model |>
    setup(
      loss = nn_mse_loss(),
      optimizer = optim_adam,
      metrics = list(luz_metric_mse())
    ) |>
    set_hparams(
      input_size = 11,
      output_size = 1
    ) |>
    fit(
      train_dl,
      epochs = epochs
    )

  # save model for this iteration
  # i.e. site left out
  luz_save(
    fitted,
    file.path(
      here::here("data/leave_site_out_weights_fcn/",
                 paste0(site, ".pt"))
    )
  )

  # run the model on the test data
  # model output sits in gpu memory,
  # export to cpu to mix with R variables
  pred <- predict(fitted, test_dl)
  pred <- (as.numeric(torch_tensor(pred, device = "cpu")))

  # add date for easy integration in
  # original data
  date <- df |>
    dplyr::filter(
      sitename == !!site
    ) |>
    dplyr::select(
      date
    )

  # return comparisons
  return(data.frame(
    sitename = site,
    date = date,
    GPP_pred = pred
  ))
})

# compile everything in a tidy dataframe
leave_site_out_output <- bind_rows(leave_site_out_output)

# save as a compressed RDS
saveRDS(
  leave_site_out_output,
  "data/leave_site_out_output_fcn.rds",
  compress = "xz"
)
