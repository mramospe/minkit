/** Definition of functions to execute in GPU arrays.
 */

/** Arange (only modifies real values)
 *
 * Reikna does not seem to handle very well complex numbers. Setting
 * "vmin" as a complex results in undefined behaviour some times.
 */
KERNEL void arange_complex( GLOBAL_MEM double2 *out, double vmin )
{
  SIZE_T idx = get_global_id(0);
  out[idx].x = vmin + idx;
  out[idx].y = 0.;
}

/// Arange
KERNEL void arange_int( GLOBAL_MEM int *out, int vmin )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = vmin + idx;
}

/// Assign values
KERNEL void assign_double( GLOBAL_MEM double *out, GLOBAL_MEM double *in, int offset )
{
  SIZE_T idx = get_global_id(0);
  out[idx + offset] = in[idx];
}

/// Assign values
KERNEL void assign_bool( GLOBAL_MEM unsigned *out, GLOBAL_MEM unsigned *in, int offset )
{
  SIZE_T idx = get_global_id(0);
  out[idx + offset] = in[idx];
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

/// Linear interpolation
KERNEL void interpolate( GLOBAL_MEM double *out, GLOBAL_MEM double *in, int n, GLOBAL_MEM double *xp, GLOBAL_MEM double *yp )
{
  SIZE_T idx = get_global_id(0);

  double x = in[idx];

  for ( int i = 0; i < n; ++i ) {

    if ( x > xp[i] )
      continue;
    else {

      if ( x == xp[i] )
	out[idx] = yp[i];
      else
	out[idx] = (yp[i - 1]*(xp[i] - x) + yp[i]*(x - xp[i - 1])) / (xp[i] - xp[i - 1]);

      break;
    }
  }
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
KERNEL void geq( GLOBAL_MEM unsigned *out, GLOBAL_MEM double *in, double v )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = ( in[idx] >= v );
}

/// Less than (for arrays)
KERNEL void ale( GLOBAL_MEM unsigned *out, GLOBAL_MEM double *in1, GLOBAL_MEM double *in2 )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = ( in1[idx] < in2[idx] );
}

/// Less than
KERNEL void le( GLOBAL_MEM unsigned *out, GLOBAL_MEM double *in, double v )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = ( in[idx] < v );
}

/// Less or equal than
KERNEL void leq( GLOBAL_MEM unsigned *out, GLOBAL_MEM double *in, double v )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = ( in[idx] <= v );
}

/// Logical and
KERNEL void logical_and( GLOBAL_MEM unsigned *out, GLOBAL_MEM unsigned *in1, GLOBAL_MEM unsigned *in2 )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = (in1[idx] == in2[idx]);
}

/// Logical and
KERNEL void logical_or( GLOBAL_MEM unsigned *out, GLOBAL_MEM unsigned *in1, GLOBAL_MEM unsigned *in2 )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = (in1[idx] || in2[idx]);
}

/// Create an array of ones
KERNEL void ones_bool( GLOBAL_MEM unsigned *out )
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

/// Get elements from an array by indices
KERNEL void slice_from_integer( GLOBAL_MEM double *out, GLOBAL_MEM double *in, GLOBAL_MEM int *indices )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = in[indices[idx]];
}

/// Create an array filled with "false" till the given index
KERNEL void false_till( GLOBAL_MEM unsigned *out, int n )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = (idx >= n);
}

/// Create an array filled with "true" till the given index
KERNEL void true_till( GLOBAL_MEM unsigned *out, int n )
{
  SIZE_T idx = get_global_id(0);
  out[idx] = (idx < n);
}

/// Create an array of zeros
KERNEL void zeros_bool( GLOBAL_MEM unsigned *out )
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
