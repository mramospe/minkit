/** Define functions that depend on the number of threads per block.
 *
 */
#define THREADS_PER_BLOCK {threads_per_block}

/// Convert a boolean array into an array of indices where the indices
/// are positioned to the begining of the block.
KERNEL void compact_indices(GLOBAL_MEM int *indices, GLOBAL_MEM int *sizes, GLOBAL_MEM unsigned *mask) {{

    int bid = get_group_id(0);
    int tid = get_local_id(0);
    int ttid = bid * THREADS_PER_BLOCK  + tid;

    unsigned val = mask[ttid];

    LOCAL_MEM unsigned cache[THREADS_PER_BLOCK];

    cache[tid] = val;

    LOCAL_BARRIER;

    if (val) {{

	int cnt = 0;
	for (int i = 0; i < tid; ++i)
	  if (cache[i])
            ++cnt;

	indices[bid * THREADS_PER_BLOCK + cnt] = ttid;
      }}

    if (tid == 0) {{
	int s = val;
	for (int i = 1; i < THREADS_PER_BLOCK; ++i)
	  if (cache[i])
            ++s;
	sizes[bid] = s;
      }}
  }}
