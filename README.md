# jumpdata

## Notes on WIP Analytics

### Jump 
- Jump status: We can approximate when the user is jumping by taking the rolling variance on load and labeling high variance windows as jumps (currently trying with \sigma > 0.2). Window size is currently 900ms but can be learned once labeled data is provided. This method is pretty robust, even with various window size.
- Load, preload, landing: We can attempt to label parts of a jump given a model for jump loads. A moving average autoregression or a hidden Markov model would require more data to be tuned. Given variation in jumps and user characteristics, we anticipate requiring some effort to get good accuracy here.
- Jump time: Calculate from continuous labels on jump status
- Jump height: Calculated by integrating g over jump time (note that this will not be accurate in environments experiencing different net ground reaction forces, i.e. space). 

### Strength
- Peak force: Taking n-tile median load (i.e. median load in the top n^-1% slice) experienced during periods labeled in jump. Currently trying with n ~ 30.
- Power: Integrate power over duration of jump (see jump status).
- Fatigue: Approximated via decline in peak force.

### Balance
- Asymmetry Index: IHMC team has not provided sufficient data to compute asymmetry index prescribed by Jordan et. al. However, their measure was pretty rudimentary so some power-weighted integral on center of pressure (mode subtracted, of course) may actually do better anyways.
- Stability: Rolling variance on center of pressure.
- Jump stability: The above, but taking a Blackman, Hamming or Parzen rolling window during periods labeled as "jump".

### Other
- Weight: From ground reaction forces (mode of load)
- Energy consumed/Calories burned: Metabolic rate inferred from weight (above) + ( max(load, 0) - mode) integrated over time.
