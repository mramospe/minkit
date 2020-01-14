/** Definition of a Polynomial PDF.
 */
KERNEL void evaluate( GLOBAL_MEM double *out, GLOBAL_MEM double *in, int n, GLOBAL_MEM double *p ) {

  SIZE_T idx  = get_global_id(0);
  double x = in[idx];

  if ( n == 0 ) {
    out[idx] = 1.;
    return;
  }

  double o = x * p[n - 1];
  for ( int i = 1; i < n; ++i )
    o = x * (o + p[n - i - 1]);

  out[idx] = o + 1.;
}
