/** Definition of a Crystal-Ball PDF.
 */
KERNEL void evaluate( GLOBAL_MEM double *out, GLOBAL_MEM double *in, double c, double s, double a, double n )
{
  SIZE_T idx = get_global_id(0);
  double x = in[idx];

  double t = ( a < 0 ? -1 : +1 ) * ( x - c ) / s;

  double aa = abs(a);

  if ( t >= -aa )
    out[idx] = exp(-0.5 * t * t);
  else {
    double A = pow(n / aa, n) * exp(-0.5 * aa * aa);
    double B = n / aa - aa;

    out[idx] = A / pow(B - t, n);
  }
}
