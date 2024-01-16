
read_data <- function(){

  if(!file.exists("data/df_imputed.rds")){
    stop("input file missing, please run the pre-processing in data-raw/")
  }

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

  if(!file.exists("data/flue_stocker18nphyt.rds")){
    stop("input file missing, please run the pre-processing in data-raw/")
  }

  # read in FLUE clusters
  flue <- readRDS("data/flue_stocker18nphyt.rds") |>
    dplyr::select(
      site,
      cluster
    ) |>
    unique() |>
    na.omit() |>
    dplyr::rename(
      'sitename' = 'site'
    )

  # merge cluster with full dataset
  df <- dplyr::left_join(df, flue) |>
    dplyr::filter(
      (cluster == "cDD" | cluster == "cGR")
    )

  # return data frame
  return(df)
}
