/// Evaluate on an unbinned data set
KERNEL void evaluate( GLOBAL_MEM double *out, GLOBAL_MEM double *in, double a, double theta, double alpha, double beta ) {

  SIZE_T idx = get_global_id(0);

  double x = in[idx];

  double d = (x - a) / theta;

  out[idx] = pow(d, alpha * beta - 1) * exp(- pow(d, beta));
}
