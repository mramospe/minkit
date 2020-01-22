// sqrt(0.5 * pi)
#define SQRTPI_2 1.2533141373155001

/// Evaluate on an unbinned data set
KERNEL void evaluate( GLOBAL_MEM double *out, GLOBAL_MEM double *in, double c, double s )
{
  SIZE_T idx = get_global_id(0);
  double x = in[idx];

  double s2 = s * s;
  double d  = (x - c);

  out[idx] = exp(-d * d / (2 * s2));
}

/// Evaluate on a binned data set
KERNEL void evaluate_binned( GLOBAL_MEM double *out, double c, double s, int nedges, int gap, GLOBAL_MEM double *edges )
{
  SIZE_T idx = get_global_id(0);

  int ie = (idx / gap) % nedges;

  double xmin = edges[ie];
  double xmax = edges[ie + 1];

  double sqrt2s = M_SQRT2 * s;
  double pmin   = (xmin - c) / sqrt2s;
  double pmax   = (xmax - c) / sqrt2s;

  out[idx] = SQRTPI_2 * s * (erf(pmax) - erf(pmin));
}
