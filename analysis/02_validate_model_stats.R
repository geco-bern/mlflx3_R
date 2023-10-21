# validate model stats

# load libraries
library(dplyr)
library(ggplot2)

# read in the data
df <- readRDS("data/leave_site_out_output.rds")

# R squared and RMSE global
statistics_global <- df |>
  summarize(
    R2 = summary(lm(GPP_pred ~ GPP_obs))$r.squared,
    RMSE = sqrt(mean((GPP_pred - GPP_obs)^2))
  )

# R squared and RMSE by site
statistics_sites <- df |>
  group_by(sitename) |>
  summarize(
    R2 = summary(lm(GPP_pred ~ GPP_obs))$r.squared,
    RMSE = sqrt(mean((GPP_pred - GPP_obs)^2))
  )

# plot all validation graphs
p <- ggplot(df) +
  geom_point(
    aes(
      GPP_obs,
      GPP_pred
    ),
    alpha = 0.2
  ) +
  labs(
    x = "GPP observed",
    y = "GPP predicted"
  ) +
  geom_abline() +
  theme_bw() +
  facet_wrap(~sitename)

print(p)

# plot all validation graphs
p <- ggplot(df) +
  geom_line(
    aes(
      1:length(GPP_obs),
      GPP_pred
    )
  ) +
  geom_line(
    aes(
      1:length(GPP_pred),
      GPP_pred
    ),
    col = "red"
  ) +
  labs(
    x = "GPP observed",
    y = "GPP predicted"
  ) +
  theme_bw() +
  facet_wrap(~sitename,
             scales = "free")

print(p)
