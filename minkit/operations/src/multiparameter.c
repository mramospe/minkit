/** Code to determine the sum of entries inside bins
 * 
 * These functions make use of parallelization in two dimensions.
 *
 */
#ifndef __CUDACC__ // Using OpenCL

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

// Needed since "atomicAdd" is not defined in OpenCL
void atomicAdd(volatile __global double *addr, double val)
{
  union {
    long u;
    double f;
  } next, expected, current;

  current.f = *addr;

  do {
    expected.f = current.f;
    next.f = expected.f + val;
    current.u = atomic_cmpxchg( (volatile __global long *) addr,
				expected.u, next.u);
  } while( current.u != expected.u );
}
#endif

// Sum the number of entries inside a bin
KERNEL void sum_inside_bins( GLOBAL_MEM double *out, int len, GLOBAL_MEM double *data, int ndim, GLOBAL_MEM int *gaps, GLOBAL_MEM double *edges ) {

  SIZE_T idx = get_global_id(0);
  SIZE_T idy = get_global_id(1);

  int r = idy;
  for ( int i = ndim - 1; i >= 0; --i ) {

    int k = r / gaps[i];

    double lb = edges[k];
    double ub = edges[k + 1];

    double v = data[len * i + idx];

    if ( v < lb || v >= ub )
      return;

    r %= gaps[i];
  }

  atomicAdd(&out[idy], 1);
}

// Sum the values inside the given bounds
KERNEL void sum_inside_bins_with_values( GLOBAL_MEM double *out, int len, GLOBAL_MEM double *data, int ndim, GLOBAL_MEM int *gaps, GLOBAL_MEM double *edges, GLOBAL_MEM double *values ) {

  SIZE_T idx = get_global_id(0);
  SIZE_T idy = get_global_id(1);

  int r = idy;
  for ( int i = ndim - 1; i >= 0; --i ) {

    int k = r / gaps[i];

    double lb = edges[k];
    double ub = edges[k + 1];

    double v = data[len * i + idx];

    if ( v < lb || v >= ub )
      return;

    r %= gaps[i];
  }

  atomicAdd(&out[idy], values[idx]);
}
