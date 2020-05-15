/** Definition of functions to execute in GPU arrays.
 */
#define INVALID_INDEX -1
#define SPLINE_ORDER 3
#define SPLINE_KNOTS_SIZE 7 // Must be 2 * SPLINE_ORDER + 1

/** Arange (only modifies real values)
 *
 * Reikna does not seem to handle very well complex numbers. Setting
 * "vmin" as a complex results in undefined behaviour some times.
 */
KERNEL void arange_complex(int lgth, GLOBAL_MEM double2 *out, double vmin) {
  int idx = get_global_id(0);

  if (idx >= lgth) // pad condition
    return;

  out[idx].x = vmin + idx;
  out[idx].y = 0.;
}

/// Arange
KERNEL void arange_int(int lgth, GLOBAL_MEM int *out, int vmin) {
  int idx = get_global_id(0);

  if (idx >= lgth) // pad condition
    return;

  out[idx] = vmin + idx;
}

// Do not consider the last elements of a data array
KERNEL void assign_double(int lgth, GLOBAL_MEM double *out, int ndim,
                          GLOBAL_MEM double *in) {
  int idx = get_global_id(0) * ndim;

  if (idx >= ndim * lgth) // pad condition
    return;

  for (int i = 0; i < ndim; ++i)
    out[i + idx] = in[i + idx];
}

/// Assign values
KERNEL void assign_double_with_offset(int lgth, GLOBAL_MEM double *out,
                                      GLOBAL_MEM double *in, int offset) {
  int idx = get_global_id(0);

  if (idx >= lgth) // pad condition
    return;

  out[idx + offset] = in[idx];
}

/// Assign values
KERNEL void assign_bool_with_offset(int lgth, GLOBAL_MEM unsigned *out,
                                    GLOBAL_MEM unsigned *in, int offset) {
  int idx = get_global_id(0);

  if (idx >= lgth) // pad condition
    return;

  out[idx + offset] = in[idx];
}

/// Exponential (complex)
KERNEL void exponential_complex(int lgth, GLOBAL_MEM double2 *out,
                                GLOBAL_MEM double2 *in) {
  int idx = get_global_id(0);

  if (idx >= lgth) // pad condition
    return;

  double2 v = in[idx];

  double d = exp(v.x);

  out[idx].x = d * cos(v.y);
  out[idx].y = d * sin(v.y);
}

/// Exponential (double)
KERNEL void exponential_double(int lgth, GLOBAL_MEM double *out,
                               GLOBAL_MEM double *in) {
  int idx = get_global_id(0);

  if (idx >= lgth) // pad condition
    return;

  out[idx] = exp(in[idx]);
}

/// Linear interpolation
KERNEL void interpolate_linear(int lgth_x, int lgth_y, GLOBAL_MEM double *out,
                               int ndim, int data_idx, GLOBAL_MEM double *in,
                               GLOBAL_MEM double *xp, GLOBAL_MEM double *yp) {

  int idx = get_global_id(0);
  int idy = get_global_id(1);

  if (idx >= lgth_x || idy >= lgth_y)
    return;

  double x = in[data_idx + idy * ndim];

  if (x == xp[idx]) { // we landed in a value of the grid
    out[idy] = yp[idx];
  } else if (idx == 0 && x < xp[idx]) { // first element
    out[idy] = 0;                       // out of range
  } else if (idx == lgth_x - 1) {

    if (x > xp[idx])
      out[idy] = 0; // out of range
  } else if (x > xp[idx] && x < xp[idx + 1]) {

    out[idy] = (yp[idx] * (xp[idx + 1] - x) + yp[idx + 1] * (x - xp[idx])) /
               (xp[idx + 1] - xp[idx]);
  } else
    ; // do nothing
}

/// Function to evaluate a B-spline
WITHIN_KERNEL double evaluate_bspline(double x, int k, int i, double *t) {

  if (k == 0)
    return (x >= t[i] && x < t[i + 1]) ? 1. : 0.;

  double c1 = (t[i + k] == t[i]) ? 0.
                                 : (x - t[i]) / (t[i + k] - t[i]) *
                                       evaluate_bspline(x, k - 1, i, t);

  double c2 = (t[i + k + 1] == t[i + 1])
                  ? 0.
                  : (t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1]) *
                        evaluate_bspline(x, k - 1, i + 1, t);

  return c1 + c2;
}

/// Interpolation using the parameters of a cubic spline (obtained from SciPy)
KERNEL void interpolate_spline(int lgth_x, int lgth_y, GLOBAL_MEM double *out,
                               int ndim, int data_idx, GLOBAL_MEM double *in,
                               GLOBAL_MEM double *knots,
                               GLOBAL_MEM double *coeffs) {

  int idx = get_global_id(0);
  int idy = get_global_id(1);

  if (idx >= lgth_x || idy >= lgth_y) // pad condition
    return;

  double x = in[data_idx + idy * ndim];

  if (x < knots[idx] || x >= knots[idx + 1]) // only the thread whose bounds
                                             // cover "x" fills the value
    return;

  // Iterate from -k to 0 to sum all the values that are different from zero
  int midx = idx > SPLINE_ORDER ? idx - SPLINE_ORDER : 0;

  double t[SPLINE_KNOTS_SIZE];
  for (int i = 0; i < SPLINE_KNOTS_SIZE; ++i)
    t[i] = knots[midx + i];

  double s = 0.;
  for (int i = 0; i <= idx - midx; ++i)
    s += coeffs[midx + i] * evaluate_bspline(x, SPLINE_ORDER, i, t);

  out[idy] = s;
}

/// Check if input data values are inside a set of bounds
KERNEL void is_inside(int lgth, GLOBAL_MEM unsigned *out, int ndim,
                      GLOBAL_MEM double *in, GLOBAL_MEM double *lb,
                      GLOBAL_MEM double *ub) {
  int idx = get_global_id(0);

  if (idx >= lgth) // pad condition
    return;

  out[idx] = true;
  for (int i = 0; i < ndim; ++i) {

    double v = in[i + idx * ndim];

    if (v < lb[i] || v >= ub[i]) {
      out[idx] = false;
      return;
    }
  }
}

/// Linspace (endpoints included)
KERNEL void linspace(int lgth, GLOBAL_MEM double *out, double vmin, double vmax,
                     int size) {
  int idx = get_global_id(0);

  if (idx >= lgth) // pad condition
    return;

  out[idx] = vmin + idx * (vmax - vmin) / (size - 1);
}

/// Logarithm
KERNEL void logarithm(int lgth, GLOBAL_MEM double *out, GLOBAL_MEM double *in) {
  int idx = get_global_id(0);

  if (idx >= lgth) // pad condition
    return;

  double x = in[idx];
  out[idx] = log(x);
}

/// Meshgrid
KERNEL void meshgrid(int lgth, GLOBAL_MEM double *out, int ndim,
                     GLOBAL_MEM int *gaps, GLOBAL_MEM double *lb,
                     GLOBAL_MEM double *steps) {
  int idx = get_global_id(0);

  if (idx >= lgth) // pad condition
    return;

  int r = idx;
  for (int i = ndim - 1; i >= 0; --i) {

    int g = gaps[i];

    out[i + ndim * idx] = lb[i] + steps[i] * (r / g);

    r %= g;
  }
}

/// Greater or equal than
KERNEL void ge(int lgth, GLOBAL_MEM unsigned *out, GLOBAL_MEM double *in,
               double v) {
  int idx = get_global_id(0);

  if (idx >= lgth) // pad condition
    return;

  out[idx] = (in[idx] >= v);
}

/// Less than (for arrays)
KERNEL void alt(int lgth, GLOBAL_MEM unsigned *out, GLOBAL_MEM double *in1,
                GLOBAL_MEM double *in2) {
  int idx = get_global_id(0);

  if (idx >= lgth) // pad condition
    return;

  out[idx] = (in1[idx] < in2[idx]);
}

/// Fill an array of indices with invalid values (-1)
KERNEL void invalid_indices(int lgth, GLOBAL_MEM int *indices) {
  int idx = get_global_id(0);

  if (idx >= lgth) // pad condition
    return;

  indices[idx] = INVALID_INDEX;
}

/// Less than
KERNEL void lt(int lgth, GLOBAL_MEM unsigned *out, GLOBAL_MEM double *in,
               double v) {
  int idx = get_global_id(0);

  if (idx >= lgth) // pad condition
    return;

  out[idx] = (in[idx] < v);
}

/// Less or equal than
KERNEL void le(int lgth, GLOBAL_MEM unsigned *out, GLOBAL_MEM double *in,
               double v) {
  int idx = get_global_id(0);

  if (idx >= lgth) // pad condition
    return;

  out[idx] = (in[idx] <= v);
}

/// Logical and
KERNEL void logical_and(int lgth, GLOBAL_MEM unsigned *out,
                        GLOBAL_MEM unsigned *in1, GLOBAL_MEM unsigned *in2) {
  int idx = get_global_id(0);

  if (idx >= lgth) // pad condition
    return;

  out[idx] = (in1[idx] == in2[idx]);
}

/// Logical and
KERNEL void logical_or(int lgth, GLOBAL_MEM unsigned *out,
                       GLOBAL_MEM unsigned *in1, GLOBAL_MEM unsigned *in2) {
  int idx = get_global_id(0);

  if (idx >= lgth) // pad condition
    return;

  out[idx] = (in1[idx] || in2[idx]);
}

/// Create an array of ones
KERNEL void ones_bool(int lgth, GLOBAL_MEM unsigned *out) {
  int idx = get_global_id(0);

  if (idx >= lgth) // pad condition
    return;

  out[idx] = true;
}

/// Create an array of ones
KERNEL void ones_double(int lgth, GLOBAL_MEM double *out) {
  int idx = get_global_id(0);

  if (idx >= lgth) // pad condition
    return;

  out[idx] = 1.;
}

/// Multiply two arrays and return zero if any of the two values is zero for a
/// given index
KERNEL void product_by_zero_is_zero(int lgth, GLOBAL_MEM double *out,
                                    GLOBAL_MEM double *in1,
                                    GLOBAL_MEM double *in2) {

  int idx = get_global_id(0);

  if (idx >= lgth)
    return;

  if (in1[idx] == 0. || in2[idx] == 0.)
    out[idx] = 0.;
  else
    out[idx] = in1[idx] * in2[idx];
}

/// Define a random grid given a set of random numbers
KERNEL void parse_random_grid(int lgth, GLOBAL_MEM double *out, int ndim,
                              GLOBAL_MEM double *lb, GLOBAL_MEM double *ub,
                              GLOBAL_MEM double *rndm) {

  int idx = get_global_id(0) * ndim;

  if (idx >= ndim * lgth) // pad condition
    return;

  for (int i = 0; i < ndim; ++i)
    out[i + idx] = lb[i] + (ub[i] - lb[i]) * rndm[i + idx];
}

/// Take the real part of an array
KERNEL void real(int lgth, GLOBAL_MEM double *out, GLOBAL_MEM double2 *in) {
  int idx = get_global_id(0);

  if (idx >= lgth) // pad condition
    return;

  out[idx] = in[idx].x;
}

/// Get elements from an array by indices
KERNEL void slice_from_integer(int lgth, GLOBAL_MEM double *out, int ndim,
                               GLOBAL_MEM double *in, GLOBAL_MEM int *indices) {
  int idx = get_global_id(0);

  if (idx >= lgth) // pad condition
    return;

  for (int i = 0; i < ndim; ++i)
    out[i + idx] = in[i + indices[idx]];
}

/// Do a sum of elements that are separated by a certain amount
KERNEL void stepped_sum(int lgth, GLOBAL_MEM double *out, int nbins,
                        int ndata_blocks, GLOBAL_MEM double *partial_sum) {

  int idx = get_global_id(0);

  if (idx >= lgth)
    return;

  double s = 0;
  for (int i = 0; i < ndata_blocks; ++i)
    s += partial_sum[i * nbins + idx];

  out[idx] = s;
}

/// Take elements from an array after the compact indices are obtained using
/// "compact_indices"
KERNEL void take(int lgth, GLOBAL_MEM double *out, int ndim,
                 GLOBAL_MEM int *sizes, GLOBAL_MEM int *indices,
                 GLOBAL_MEM double *in) {

  int ttid = get_global_id(0);

  if (ttid >= lgth)
    return;

  int val = indices[ttid];

  if (val == INVALID_INDEX)
    return;

  int bid = get_group_id(0);
  int step = get_local_id(0);
  for (int i = 0; i < bid; ++i)
    step += sizes[i];

  step *= ndim;
  val *= ndim;

  for (int i = 0; i < ndim; ++i)
    out[i + step] = in[i + val];
}

/// Take elements separated by a certain amount
KERNEL void take_each(int lgth, GLOBAL_MEM double *out, GLOBAL_MEM double *in,
                      int each, int start) {

  int idx = get_global_id(0);

  if (idx >= lgth)
    return;

  out[idx] = in[start + idx * each];
}

/// Take elements in a range
KERNEL void take_slice(GLOBAL_MEM double *out, GLOBAL_MEM double *in, int start,
                       int end) {

  int idx = get_global_id(0);

  if (idx >= (end - start))
    return;

  out[idx] = in[idx + start];
}

/// Create an array of zeros
KERNEL void zeros_bool(int lgth, GLOBAL_MEM unsigned *out) {
  int idx = get_global_id(0);

  if (idx >= lgth) // pad condition
    return;

  out[idx] = false;
}

/// Create an array of zeros
KERNEL void zeros_double(int lgth, GLOBAL_MEM double *out) {
  int idx = get_global_id(0);

  if (idx >= lgth) // pad condition
    return;

  out[idx] = 0.;
}
