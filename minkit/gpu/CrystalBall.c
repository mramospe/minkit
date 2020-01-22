// sqrt(0.5 * pi)
#define SQRTPI_2 1.2533141373155001
// sqrt(2)
#define SQRT2 1.4142135623730951

/// Evaluate on an unbinned data set
KERNEL void evaluate( GLOBAL_MEM double *out, GLOBAL_MEM double *in, double c, double s, double a, double n )
{
  SIZE_T idx = get_global_id(0);
  double x = in[idx];

  double t = ( a < 0 ? -1 : +1 ) * ( x - c ) / s;

  double aa = fabs(a);

  if ( t >= -aa )
    out[idx] = exp(-0.5 * t * t);
  else {
    double A = pow(n / aa, n) * exp(-0.5 * aa * aa);
    double B = n / aa - aa;

    out[idx] = A / pow(B - t, n);
  }
}

/// Evaluate on a binned data set
KERNEL void evaluate_binned( GLOBAL_MEM double *out, double c, double s, double a, double n, int nedges, int gap, GLOBAL_MEM double *edges )
{
  SIZE_T idx = get_global_id(0);

  int ie = (idx / gap) % nedges;

  double xmin = edges[ie];
  double xmax = edges[ie + 1];

  double result = 0.;

  bool use_log = ( fabs(n - 1.) < 1e-5 );

  double tmin = (xmin - c) / s;
  double tmax = (xmax - c) / s;

  if ( a < 0 ) {

    double tmp = tmin;
    tmin = -tmax;
    tmax = -tmp;
  }

  double aa = fabs(a);

  if ( tmin >= -aa )
    result += s * SQRTPI_2 * (erf(tmax / SQRT2) - erf(tmin / SQRT2));
  else {
    double A = pow(n / aa, n) * exp(-0.5 * aa * aa);
    double B = n / aa - aa;

    if ( tmax <= -aa ) {

      if ( use_log )
	result += A * s * (log(B - tmin) - log(B - tmax));
      else
	result += A * s / (1. - n) * (1. / pow(B - tmin, n - 1.) - 1. / pow(B - tmax, n - 1.));
    }
    else {
      double t1 = 0.;

      if ( use_log )
	t1 = A * s * (log(B - tmin) - log(n / aa));
      else
	t1 = A * s / (1. - n) * (1. / pow(B - tmin, n - 1.) - 1. / pow(n / aa, n - 1.));

      double t2 = s * SQRTPI_2 * (erf(tmax / SQRT2) - erf(-aa / SQRT2));

      result += t1 + t2;
    }
  }

  out[idx] = result != 0. ? result : 1e-300;
}
