# Impute data
#
# Deals with missing values in some of the variables
# by using KNN imputation

library(recipes)
library(dplyr)

#---- read in the raw data ----
df <- readr::read_csv("data-raw/df_20210510.csv") |>
  dplyr::filter(
    sitename != "CN-Cng"
  ) |>
  dplyr::select(
    -'lat',
    -'lon',
    -'elv',
    -'c4',
    -'whc',
    -'LE_F_MDS',
    -'NEE_VUT_REF'
  )

#---- flagging filled locations ----

# assumes that they will be filled
# below
na_values <- is.na(df)
na_values <- apply(na_values, 1, any)
df$knn_fill <- na_values

#---- Impute TA_F_DAY -----
message("Imputing TA_F_DAY values...")
df <- df |>
  group_by(sitename) |>
  do({

    # Impute TA_F_DAY with TA_F and SW_IN_F
    df_TA <- . |>
      dplyr::select(
        'TA_F',
        'SW_IN_F',
        'TA_F_DAY',
        'TA_F_NIGHT'
      )

    # impute missing with KNN
    pp <- recipes::recipe(
      TA_F_DAY ~ TA_F + SW_IN_F,
      data = df_TA
    ) |>
      recipes::step_center(
        recipes::all_numeric(),
        -recipes::all_outcomes()
      ) |>
      recipes::step_scale(
        recipes::all_numeric(),
        -recipes::all_outcomes()
      ) |>
      recipes::step_impute_knn(
        recipes::all_outcomes(),
        neighbors = 5
      )

    pp_prep <- recipes::prep(
      pp,
      training = df_TA
    )

    df_baked <- recipes::bake(
      pp_prep,
      new_data = .
    )

    # fill missing with gap-filled
    output <- . |>
      dplyr::bind_cols(
        df_baked |>
          dplyr::select(
            TA_F_DAY_FILLED = TA_F_DAY
          )
      ) |>
      dplyr::mutate(
        TA_F_DAY = ifelse(
          is.na(TA_F_DAY),
          TA_F_DAY_FILLED,
          TA_F_DAY)
      ) |>
      dplyr::select(
        -TA_F_DAY_FILLED
      )

     output
  }) |>
  ungroup()

#---- Impute TA_F_NIGHT -----
message("Imputing TA_F_NIGHT values...")
df <- df |>
  group_by(sitename) |>
  do({

    # Impute TA_F_DAY with TA_F and SW_IN_F
    df_TA <- . |>
      dplyr::select(
        'TA_F',
        'SW_IN_F',
        'TA_F_DAY',
        'TA_F_NIGHT'
      )

    # impute missing with KNN
    pp <- recipes::recipe(
      TA_F_NIGHT ~ TA_F + SW_IN_F,
      data = df_TA
    ) |>
      recipes::step_center(
        recipes::all_numeric(),
        -recipes::all_outcomes()
      ) |>
      recipes::step_scale(
        recipes::all_numeric(),
        -recipes::all_outcomes()
      ) |>
      recipes::step_impute_knn(
        recipes::all_outcomes(),
        neighbors = 5
      )

    pp_prep <- recipes::prep(
      pp,
      training = df_TA
    )

    df_baked <- recipes::bake(
      pp_prep,
      new_data = .
    )

    # fill missing with gap-filled
    output <- . |>
      dplyr::bind_cols(
        df_baked |>
          dplyr::select(
            TA_F_NIGHT_FILLED = TA_F_NIGHT
          )
      ) |>
      dplyr::mutate(
        TA_F_NIGHT = ifelse(
          is.na(TA_F_NIGHT),
          TA_F_NIGHT_FILLED,
          TA_F_NIGHT)
      ) |>
      dplyr::select(
        -TA_F_NIGHT_FILLED
      )

    output
  }) |>
  ungroup()

#---- Impute GPP -----
message("Imputing GPP values...")
df <- df |>
  group_by(sitename) |>
  do({

    # Impute GPP
    df_GPP <- . |>
      dplyr::select(
        'TA_F',
        'SW_IN_F',
        'TA_F_DAY',
        'LW_IN_F',
        'WS_F',
        'P_F',
        'VPD_F',
        'GPP_NT_VUT_REF'
      )

    # impute missing with KNN
    pp <- recipes::recipe(
      GPP_NT_VUT_REF ~ .,
      data = df_GPP
    ) |>
      recipes::step_center(
        recipes::all_numeric(),
        -recipes::all_outcomes()
      ) |>
      recipes::step_scale(
        recipes::all_numeric(),
        -recipes::all_outcomes()
      ) |>
      recipes::step_impute_knn(
        recipes::all_outcomes(),
        neighbors = 5
      )

    pp_prep <- recipes::prep(
      pp,
      training = df_GPP
    )

    df_baked <- recipes::bake(
      pp_prep,
      new_data = .
    )

    # fill missing with gap-filled
    output <- . |>
      dplyr::bind_cols(
        df_baked |>
          dplyr::select(
            GPP_NT_VUT_REF_FILLED = GPP_NT_VUT_REF
          )
      ) |>
      dplyr::mutate(
        GPP_NT_VUT_REF = ifelse(
          is.na(GPP_NT_VUT_REF),
          GPP_NT_VUT_REF_FILLED,
          GPP_NT_VUT_REF
          )
      ) |>
      dplyr::select(
        -GPP_NT_VUT_REF_FILLED
      )
    output
  }) |>
  ungroup()

#---- saving data ----

saveRDS(df, "data/df_imputed.rds", compress = "xz")
