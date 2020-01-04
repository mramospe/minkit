/** Definition of a Gaussian PDF.
 */
KERNEL void evaluate( GLOBAL_MEM double *out, GLOBAL_MEM double *in, double c, double s )
{
  SIZE_T idx = get_global_id(0);
  double x = in[idx];

  double s2 = s * s;
  double d  = (x - c);

  out[idx] = exp(-d * d / (2 * s2));
}
