# forward index to subset data based upon
# index, optionally sample the sites further
# to leave a fraction of the sites
self$idx <- sort(sample.int(
  n = length(self$sites),
  size = length(self$sites) * sample_frac
))
