/** Definition of functions to execute in GPU arrays.
 */

/** Arange (only modifies real values)
 *
 * Reikna does not seem to handle very well complex numbers. Setting
 * "vmin" as a complex results in undefined behaviour some times.
 */
KERNEL void arange_complex(GLOBAL_MEM double2 *out, double vmin) {
  SIZE_T idx = get_global_id(0);
  out[idx].x = vmin + idx;
  out[idx].y = 0.;
}

/// Arange
KERNEL void arange_int(GLOBAL_MEM int *out, int vmin) {
  SIZE_T idx = get_global_id(0);
  out[idx] = vmin + idx;
}

/// Assign values
KERNEL void assign_double(GLOBAL_MEM double *out, GLOBAL_MEM double *in,
                          int offset) {
  SIZE_T idx = get_global_id(0);
  out[idx + offset] = in[idx];
}

/// Assign values
KERNEL void assign_bool(GLOBAL_MEM unsigned *out, GLOBAL_MEM unsigned *in,
                        int offset) {
  SIZE_T idx = get_global_id(0);
  out[idx + offset] = in[idx];
}

/// Exponential (complex)
KERNEL void exponential_complex(GLOBAL_MEM double2 *out,
                                GLOBAL_MEM double2 *in) {
  SIZE_T idx = get_global_id(0);
  double2 v = in[idx];

  double d = exp(v.x);

  out[idx].x = d * cos(v.y);
  out[idx].y = d * sin(v.y);
}

/// Exponential (double)
KERNEL void exponential_double(GLOBAL_MEM double *out, GLOBAL_MEM double *in) {
  SIZE_T idx = get_global_id(0);
  double x = in[idx];

  out[idx] = exp(x);
}

// Do not consider the last elements of a data array
KERNEL void keep_to_limit(GLOBAL_MEM double *out, int maximum, int ndim,
                          int len, GLOBAL_MEM double *in) {
  SIZE_T idx = get_global_id(0);

  for (int i = 0; i < ndim; ++i)
    out[i * maximum + idx] = in[i * len + idx];
}

/// Linear interpolation
KERNEL void interpolate(GLOBAL_MEM double *out, int ndim, int len,
                        GLOBAL_MEM int *data_idx, GLOBAL_MEM double *in, int n,
                        GLOBAL_MEM double *xp, GLOBAL_MEM double *yp) {

  SIZE_T idx = get_global_id(0);

  double x = in[data_idx[0] * len + idx];

  for (int i = 0; i < n; ++i) {

    if (x > xp[i])
      continue;
    else {

      if (x == xp[i])
        out[idx] = yp[i];
      else
        out[idx] = (yp[i - 1] * (xp[i] - x) + yp[i] * (x - xp[i - 1])) /
                   (xp[i] - xp[i - 1]);

      break;
    }
  }
}

KERNEL void is_inside(GLOBAL_MEM unsigned *out, int len, GLOBAL_MEM double *in,
                      int ndim, GLOBAL_MEM double *lb, GLOBAL_MEM double *ub) {
  SIZE_T idx = get_global_id(0);
  out[idx] = true;
  for (int i = 0; i < ndim; ++i) {
    double v = in[i * len + idx];
    if (v < lb[i] || v >= ub[i]) {
      out[idx] = false;
      return;
    }
  }
}

/// Linspace (endpoints included)
KERNEL void linspace(GLOBAL_MEM double *out, double vmin, double vmax,
                     int size) {
  SIZE_T idx = get_global_id(0);
  out[idx] = vmin + idx * (vmax - vmin) / (size - 1);
}

/// Logarithm
KERNEL void logarithm(GLOBAL_MEM double *out, GLOBAL_MEM double *in) {
  SIZE_T idx = get_global_id(0);
  double x = in[idx];
  out[idx] = log(x);
}

/// Greater or equal than
KERNEL void ge(GLOBAL_MEM unsigned *out, GLOBAL_MEM double *in, double v) {
  SIZE_T idx = get_global_id(0);
  out[idx] = (in[idx] >= v);
}

/// Less than (for arrays)
KERNEL void alt(GLOBAL_MEM unsigned *out, GLOBAL_MEM double *in1,
                GLOBAL_MEM double *in2) {
  SIZE_T idx = get_global_id(0);
  out[idx] = (in1[idx] < in2[idx]);
}

/// Fill an array of indices with invalid values (-1)
KERNEL void invalid_indices(GLOBAL_MEM int *indices) {

  int idx = get_global_id(0);
  indices[idx] = -1;
}


/// Less than
KERNEL void lt(GLOBAL_MEM unsigned *out, GLOBAL_MEM double *in, double v) {
  SIZE_T idx = get_global_id(0);
  out[idx] = (in[idx] < v);
}

/// Less or equal than
KERNEL void le(GLOBAL_MEM unsigned *out, GLOBAL_MEM double *in, double v) {
  SIZE_T idx = get_global_id(0);
  out[idx] = (in[idx] <= v);
}

/// Logical and
KERNEL void logical_and(GLOBAL_MEM unsigned *out, GLOBAL_MEM unsigned *in1,
                        GLOBAL_MEM unsigned *in2) {
  SIZE_T idx = get_global_id(0);
  out[idx] = (in1[idx] == in2[idx]);
}

/// Logical and
KERNEL void logical_or(GLOBAL_MEM unsigned *out, GLOBAL_MEM unsigned *in1,
                       GLOBAL_MEM unsigned *in2) {
  SIZE_T idx = get_global_id(0);
  out[idx] = (in1[idx] || in2[idx]);
}

/// Create an array of ones
KERNEL void ones_bool(GLOBAL_MEM unsigned *out) {
  SIZE_T idx = get_global_id(0);
  out[idx] = true;
}

/// Create an array of ones
KERNEL void ones_double(GLOBAL_MEM double *out) {
  SIZE_T idx = get_global_id(0);
  out[idx] = 1.;
}

/// Take the real part of an array
KERNEL void real(GLOBAL_MEM double *out, GLOBAL_MEM double2 *in) {
  SIZE_T idx = get_global_id(0);
  out[idx] = in[idx].x;
}

/// Get elements from an array by indices
KERNEL void slice_from_integer(GLOBAL_MEM double *out, int dim, int len_in,
                               GLOBAL_MEM double *in, int len_indices,
                               GLOBAL_MEM int *indices) {
  SIZE_T idx = get_global_id(0);
  for (int j = 0; j < dim; ++j)
    out[j * len_indices + idx] = in[j * len_in + indices[idx]];
}

/// Take elements from an array after the compact indices are obtained using "compact_indices"
KERNEL void take(GLOBAL_MEM double *out, int dim, int lgth, GLOBAL_MEM int *sizes, GLOBAL_MEM int *indices, GLOBAL_MEM double* in) {

  int bid = get_group_id(0);
  int tid = get_local_id(0);
  int ttid = get_global_id(0);

  int val = indices[ttid];

  if (val < 0)
    return;

  int step = tid;
  for ( int i = 0; i < bid; ++i )
    step += sizes[i];

  for ( int i = 0; i < dim; ++i )
    out[i * lgth + step] = in[i * lgth + val];
}

/// Create an array filled with "false" till the given index
KERNEL void false_till(GLOBAL_MEM unsigned *out, int n) {
  SIZE_T idx = get_global_id(0);
  out[idx] = (idx >= n);
}

/// Create an array filled with "true" till the given index
KERNEL void true_till(GLOBAL_MEM unsigned *out, int n) {
  SIZE_T idx = get_global_id(0);
  out[idx] = (idx < n);
}

/// Create an array of zeros
KERNEL void zeros_bool(GLOBAL_MEM unsigned *out) {
  SIZE_T idx = get_global_id(0);
  out[idx] = false;
}

/// Create an array of zeros
KERNEL void zeros_double(GLOBAL_MEM double *out) {
  SIZE_T idx = get_global_id(0);
  out[idx] = 0.;
}
