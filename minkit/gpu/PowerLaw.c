/** Definition of the evaluation of an Exponential PDF.
 */
KERNEL void evaluate( GLOBAL_MEM double *out, GLOBAL_MEM double *in, double c, double n )
{
  SIZE_T idx  = get_global_id(0);
  double x = in[idx];

  out[idx] = 1. / pow(abs(x - c), n);
}
