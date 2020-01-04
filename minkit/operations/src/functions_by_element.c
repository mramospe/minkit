/** Definition of functions to execute in GPU arrays.
 */

/// Arange (only modifies real values)
KERNEL void arange_complex( GLOBAL_MEM double2 *out, double2 vmin )
{
  SIZE_T idx = get_global_id(0);
  out[idx].x = vmin.x + idx;
  out[idx].y = vmin.y + 0.;
}

/// Arange
KERNEL void arange_int( GLOBAL_MEM int *out, int vmin )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = vmin + idx;
}

/// Exponential (complex)
KERNEL void exponential_complex( GLOBAL_MEM double2 *out, GLOBAL_MEM double2 *in )
{
  SIZE_T idx = get_global_id(0);
  double2 v = in[idx];

  double d = exp(v.x);

  out[idx].x = d * cos(v.y);
  out[idx].y = d * sin(v.y);
}

/// Exponential (double)
KERNEL void exponential_double( GLOBAL_MEM double *out, GLOBAL_MEM double *in )
{
  SIZE_T idx = get_global_id(0);
  double x = in[idx];

  out[idx] = exp(x);
}

/// Linspace (endpoints included)
KERNEL void linspace( GLOBAL_MEM double *out, double vmin, double vmax, int size )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = vmin + idx * (vmax - vmin) / (size - 1);
}

/// Logarithm
KERNEL void logarithm( GLOBAL_MEM double *out, GLOBAL_MEM double *in )
{
  SIZE_T idx = get_global_id(0);
  double x = in[idx];
  out[idx] = log(x);
}

/// Greater or equal than
KERNEL void geq( GLOBAL_MEM bool *out, GLOBAL_MEM double *in, double v )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = ( in[idx] >= v );
}

/// Less or equal than
KERNEL void leq( GLOBAL_MEM bool *out, GLOBAL_MEM double *in, double v )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = ( in[idx] <= v );
}

/// Logical and
KERNEL void logical_and( GLOBAL_MEM bool *out, GLOBAL_MEM bool *in1, GLOBAL_MEM bool *in2 )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = (in1[idx] == in2[idx]);
}

/// Logical and
KERNEL void logical_or( GLOBAL_MEM bool *out, GLOBAL_MEM bool *in1, GLOBAL_MEM bool *in2 )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = (in1[idx] || in2[idx]);
}

/// Create an array of ones
KERNEL void ones_bool( GLOBAL_MEM bool *out )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = true;
}

/// Create an array of ones
KERNEL void ones_double( GLOBAL_MEM double *out )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = 1.;
}

/// Take the real part of an array
KERNEL void real( GLOBAL_MEM double *out, GLOBAL_MEM double2 *in )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = in[idx].x;
}

/// Create an array filled with "true" till the given index
KERNEL void true_till( GLOBAL_MEM bool *out, int n )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = (idx < n);
}

/// Create an array of zeros
KERNEL void zeros_bool( GLOBAL_MEM bool *out )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = false;
}

/// Create an array of zeros
KERNEL void zeros_double( GLOBAL_MEM double *out )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = 0.;
}
