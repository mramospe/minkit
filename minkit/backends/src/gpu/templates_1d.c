/** Define functions that depend on the number of threads per block.
 *
 */
#define THREADS_PER_BLOCK $threads_per_block

/// Convert a boolean array into an array of indices where the indices
/// are positioned to the begining of the block.
KERNEL void compact_indices(int lgth, GLOBAL_MEM int *indices,
                            GLOBAL_MEM int *sizes, GLOBAL_MEM unsigned *mask) {

  int bid = get_group_id(0);
  int tid = get_local_id(0);
  int ttid = get_global_id(0);

  LOCAL_MEM unsigned cache[THREADS_PER_BLOCK];

  if (ttid < lgth)
    cache[tid] = mask[ttid];

  LOCAL_BARRIER;

  if (ttid < lgth) { // allowed by padding

    if (cache[tid]) {

      int cnt = 0;
      for (int i = 0; i < tid; ++i)
        if (cache[i])
          ++cnt;

      indices[bid * THREADS_PER_BLOCK + cnt] = ttid;
    }
  }

  if (tid == 0 && ttid < lgth) {

    int s = 0;

    for (int i = 0; i < THREADS_PER_BLOCK; ++i) {

      if (bid * THREADS_PER_BLOCK + i >= lgth)
        break;

      if (cache[i])
        ++s;
    }

    sizes[bid] = s;
  }
}
