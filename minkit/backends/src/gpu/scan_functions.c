/****************************************
 * MIT License
 *
 * Copyright (c) 2020 Miguel Ramos Pernas
 ****************************************/

/** Define functions to scan or reduce arrays
 *
 */
#define THREADS_PER_BLOCK $threads_per_block

#define MIN_DOUBLE -1.7976931348623157e+308

/// Calculate the index of the maximum value
KERNEL void scan_max(int niter, int length, GLOBAL_MEM int *indices,
                     GLOBAL_MEM double *values, int step) {

  int gid = get_global_id(0) * niter * step;
  int tid = get_local_id(0);

  LOCAL_MEM int idx_cache[THREADS_PER_BLOCK];
  LOCAL_MEM double value_cache[THREADS_PER_BLOCK];

  value_cache[tid] = MIN_DOUBLE;

  if (gid < length) {
    for (int i = 0; i < niter; ++i) {

      if (gid + i * step >= length)
        break;

      int mid = indices[gid + i * step];

      if (values[mid] > value_cache[tid]) {
        value_cache[tid] = values[mid];
        idx_cache[tid] = mid;
      }
    }
  }

  LOCAL_BARRIER;

  if (tid == 0) { // First thread sets the maximum of the group

    int bid = get_group_id(0);

    int best = 0;
    double max = value_cache[best];

    for (int i = 1; i < THREADS_PER_BLOCK; ++i) {

      if (value_cache[i] > max) {
        best = i;
        max = value_cache[i];
      }
    }

    indices[gid] = best;
  }
}
