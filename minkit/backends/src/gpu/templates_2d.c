/** Define functions that depend on the number of threads per block in two
 * dimensions.
 *
 */
#define THREADS_PER_BLOCK_X $threads_per_block_x
#define THREADS_PER_BLOCK_Y $threads_per_block_y

// Sum the number of entries inside a bin
KERNEL void sum_inside_bins(int lgth, int nbins, GLOBAL_MEM double *out,
                            int ndim, GLOBAL_MEM double *data,
                            GLOBAL_MEM int *gaps, GLOBAL_MEM double *edges) {

  int tid_x = get_local_id(0);
  int tid_y = get_local_id(1);

  int bid_x = get_group_id(0);
  int bid_y = get_group_id(1);

  int idx = get_global_id(0);
  int idy = get_global_id(1);

  LOCAL_MEM int cache_bin[THREADS_PER_BLOCK_X];

  if (tid_y == 0) // to correctly process values out of range
    cache_bin[tid_x] = -1;

  LOCAL_BARRIER;

  if (idx < lgth && idy < nbins) {

    unsigned add = true;

    int r = idy;
    for (int i = ndim - 1; i >= 0; --i) {

      int k = r / gaps[i];

      double lb = edges[k];
      double ub = edges[k + 1];

      double v = data[i + idx * ndim];

      if (v < lb || v >= ub) {
        add = false;
        break;
      }

      r %= gaps[i];
    }

    if (add)
      cache_bin[tid_x] = tid_y; // this is the only thread setting the value
  }

  LOCAL_BARRIER;

  if (tid_x == 0 && idx < lgth && idy < nbins) {

    int s = 0;

    for (int i = 0; i < THREADS_PER_BLOCK_X; ++i)
      if (cache_bin[i] == tid_y)
        ++s;

    out[idy + bid_x * nbins] = s;
  }
}

// Sum the values inside the given bounds
KERNEL void sum_inside_bins_with_values(int lgth, int nbins,
                                        GLOBAL_MEM double *out, int ndim,
                                        GLOBAL_MEM double *data,
                                        GLOBAL_MEM int *gaps,
                                        GLOBAL_MEM double *edges,
                                        GLOBAL_MEM double *values) {

  int tid_x = get_local_id(0);
  int tid_y = get_local_id(1);

  int bid_x = get_group_id(0);
  int bid_y = get_group_id(1);

  int idx = get_global_id(0);
  int idy = get_global_id(1);

  LOCAL_MEM int cache_bin[THREADS_PER_BLOCK_X];

  if (tid_y == 0) // to correctly process values out of range
    cache_bin[tid_x] = -1;

  LOCAL_BARRIER;

  if (idx < lgth && idy < nbins) {

    unsigned add = true;

    int r = idy;
    for (int i = ndim - 1; i >= 0; --i) {

      int k = r / gaps[i];

      double lb = edges[k];
      double ub = edges[k + 1];

      double v = data[i + idx * ndim];

      if (v < lb || v >= ub) {
        add = false;
        break;
      }

      r %= gaps[i];
    }

    if (add)
      cache_bin[tid_x] = tid_y; // this is the only thread setting the value
  }

  LOCAL_BARRIER;

  if (tid_x == 0 && idx < lgth && idy < nbins) {

    double s = 0;

    for (int i = 0; i < THREADS_PER_BLOCK_X; ++i)
      if (cache_bin[i] == tid_y)
        s += values[bid_x * THREADS_PER_BLOCK_X + i];

    out[idy + bid_x * nbins] = s;
  }
}
