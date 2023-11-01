# Feedback on startup
message("Leave-Site-Out routine, generating data...")

# set both the R seed
# and the torch seed (both function independently)
set.seed(1)
torch::torch_manual_seed(42)

# required libraries
library(torch)
library(luz)
library(dplyr)
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

  # some feedback
  message(sprintf("Processing: %s", site))

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
  # for test data
  test_ds <- df |>
    dplyr::filter(
      sitename == !!site
    ) |>
    gpp_dataset(
      train_center
    )

  test_dl <- dataloader(
    test_ds,
    batch_size = 1,
    shuffle = FALSE
  )

  fitted <- try(luz_load(
    file.path(
        here::here("data/leave_site_out_weights/",
                 paste0(site, ".pt"))
    )
  ))

  if(inherits(fitted, 'try-error')){
    return(NULL)
  }

  # run the model on the test data
  pred <- predict(fitted, test_dl)

  # back convert centered data (should not be necessary update runs)
  train_mean <- train_center$GPP_NT_VUT_REF_mean
  train_sd <- train_center$GPP_NT_VUT_REF_sd
  pred <- (as.numeric(torch_tensor(pred, device = "cpu")) * train_sd) + train_mean

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
    GPP_pred = pred,
    GPP_mean = train_mean,
    GPP_sd = train_sd
  ))
})

# compile everything in a tidy dataframe
leave_site_out_output <- bind_rows(leave_site_out_output)

# save as a compressed RDS
saveRDS(
  leave_site_out_output,
  "data/leave_site_out_output.rds",
  compress = "xz"
)
