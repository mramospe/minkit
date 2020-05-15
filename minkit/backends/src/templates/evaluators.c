/******************************************************************************
 * Implementation of the main functions to evaluate PDFs.
 *
 * The following macros need to be defined before including this file:
 * - NDIM: number of dimensions the PDF will be working on
 * - NPARS: number of parameters
 * - NVAR_ARG_PARS (optional): number of variable parameters
 * - FUNCTION: function to evaluate the PDF.
 * - INTEGRAL (optional): function to calculate the integral between certain
 *   bounds.
 ******************************************************************************/
// Define the accessors (allows for a maximum of 50 arguments)
#define GETARG_1(X) X[0]
#define GETARG_2(X) GETARG_1(X), X[1]
#define GETARG_3(X) GETARG_2(X), X[2]
#define GETARG_4(X) GETARG_3(X), X[3]
#define GETARG_5(X) GETARG_4(X), X[4]
#define GETARG_6(X) GETARG_5(X), X[5]
#define GETARG_7(X) GETARG_6(X), X[6]
#define GETARG_8(X) GETARG_7(X), X[7]
#define GETARG_9(X) GETARG_8(X), X[8]
#define GETARG_10(X) GETARG_9(X), X[9]
#define GETARG_11(X) GETARG_10(X), X[10]
#define GETARG_12(X) GETARG_11(X), X[11]
#define GETARG_13(X) GETARG_12(X), X[12]
#define GETARG_14(X) GETARG_13(X), X[13]
#define GETARG_15(X) GETARG_14(X), X[14]
#define GETARG_16(X) GETARG_15(X), X[15]
#define GETARG_17(X) GETARG_16(X), X[16]
#define GETARG_18(X) GETARG_17(X), X[17]
#define GETARG_19(X) GETARG_18(X), X[18]
#define GETARG_20(X) GETARG_19(X), X[19]
#define GETARG_21(X) GETARG_20(X), X[20]
#define GETARG_22(X) GETARG_21(X), X[21]
#define GETARG_23(X) GETARG_22(X), X[22]
#define GETARG_24(X) GETARG_23(X), X[23]
#define GETARG_25(X) GETARG_24(X), X[24]
#define GETARG_26(X) GETARG_25(X), X[25]
#define GETARG_27(X) GETARG_26(X), X[26]
#define GETARG_28(X) GETARG_27(X), X[27]
#define GETARG_29(X) GETARG_28(X), X[28]
#define GETARG_30(X) GETARG_29(X), X[29]
#define GETARG_31(X) GETARG_30(X), X[30]
#define GETARG_32(X) GETARG_31(X), X[31]
#define GETARG_33(X) GETARG_32(X), X[32]
#define GETARG_34(X) GETARG_33(X), X[33]
#define GETARG_35(X) GETARG_34(X), X[34]
#define GETARG_36(X) GETARG_35(X), X[35]
#define GETARG_37(X) GETARG_36(X), X[36]
#define GETARG_38(X) GETARG_37(X), X[37]
#define GETARG_39(X) GETARG_38(X), X[38]
#define GETARG_40(X) GETARG_39(X), X[39]
#define GETARG_41(X) GETARG_40(X), X[40]
#define GETARG_42(X) GETARG_41(X), X[41]
#define GETARG_43(X) GETARG_42(X), X[42]
#define GETARG_44(X) GETARG_43(X), X[43]
#define GETARG_45(X) GETARG_44(X), X[44]
#define GETARG_46(X) GETARG_45(X), X[45]
#define GETARG_47(X) GETARG_46(X), X[46]
#define GETARG_48(X) GETARG_47(X), X[47]
#define GETARG_49(X) GETARG_48(X), X[48]
#define GETARG_50(X) GETARG_49(X), X[49]

//#define GETARG_EXPAND(...) __VA_ARGS__
#define GETARG_EXPAND(ARG) ARG

#define GETARG___N(N, X) GETARG_EXPAND(GETARG_##N)(X)
#define GETARG__N(N, X) GETARG___N(N, X)
#define GETARG_N(N, X) GETARG__N(GETARG_EXPAND(N), X)

// Build the accessors to the data and parameter values
#define DATA(X) GETARG_N(NDIM, X)
#define PARAMS(X) GETARG_N(NPARS, X)

#ifdef USE_CPU
// So we can load it from python
extern "C" {
#endif // USE_CPU

// All these macros must be used in conjunction
#ifdef NVAR_ARG_PARS
// Modify the arguments to the function ()
#if NPARS > 0
#define FWD_PARAMS(params) PARAMS(params), NVAR_ARG_PARS, params + NPARS
#else
#define FWD_PARAMS(params) NVAR_ARG_PARS, params
#endif // NPARS > 0
#else
#define FWD_PARAMS(params) PARAMS(params)
#endif // NVAR_ARG_PARS

#ifdef USE_CPU
// Define the function to evaluate on single values
double function(double *data, double *params) {

  return FUNCTION(DATA(data), FWD_PARAMS(params));
}

#ifdef INTEGRAL
// Define the integral function
double integral(double *lower_bounds, double *upper_bounds, double *params) {

  return INTEGRAL(DATA(lower_bounds), DATA(upper_bounds), FWD_PARAMS(params));
}
#endif // INTEGRAL
#endif // USE_CPU

// Evaluation on a data sample
#define EVALUATE_CODE(i, output_array, ndim, data_idx, data_array, params)     \
  {                                                                            \
    double data[NDIM];                                                         \
    for (int j = 0; j < NDIM; ++j)                                             \
      data[j] = data_array[data_idx[j] + i * ndim];                            \
                                                                               \
    output_array[i] = FUNCTION(DATA(data), FWD_PARAMS(params));                \
  }

#ifdef USE_CPU
void evaluate(int len, double *output_array, int ndim, int *data_idx,
              double *data_array, double *params) {

  for (int i = 0; i < len; ++i) {
    EVALUATE_CODE(i, output_array, ndim, data_idx, data_array, params);
  }
}
#else
KERNEL void evaluate(int lgth, GLOBAL_MEM double *output_array, int ndim,
                     GLOBAL_MEM int *data_idx, GLOBAL_MEM double *data_array,
                     GLOBAL_MEM double *params) {

  SIZE_T i = get_global_id(0);

  if (i >= lgth)
    return;

  EVALUATE_CODE(i, output_array, ndim, data_idx, data_array, params);
}
#endif // USE_CPU

/// Fill the lower and upper bounds for a given bin index
WITHIN_KERNEL void fill_bounds(double *lb, double *ub, int r, int ngaps,
                               GLOBAL_MEM int *gaps_idx, GLOBAL_MEM int *gaps,
                               GLOBAL_MEM double *edges) {

  for (int j = ngaps - 1; j >= 0; --j) {

    int k = r / gaps[j];

    for (int s = 0; s < NDIM; ++s) {

      if (gaps_idx[s] == j) {
        lb[s] = edges[k];
        ub[s] = edges[k + 1];
        break;
      }
    }

    r %= gaps[j];
  }
}

/// Calculate the integer power of another integer
WITHIN_KERNEL int int_power(int n, int p) {

  if (p == 0)
    return 1;

  int r = n;
  for (int i = 1; i < p; ++i)
    r *= n;

  return r;
}

/// Simpson's rule for numerical integration (N must be even)
#ifdef USE_CPU
WITHIN_KERNEL double simpson_rule(int N, int nsteps, double *lb, double *ub,
                                  double *params) {
#else
WITHIN_KERNEL double simpson_rule(int N, int nsteps, double *lb, double *ub,
                                  GLOBAL_MEM double *params) {
#endif // USE_CPU

  const double p2 = int_power(2, NDIM);

  // Define the steps and the global factor for the integral
  double I = 1. / p2;

  double steps[NDIM];
  for (int i = 0; i < NDIM; ++i) {

    steps[i] = (ub[i] - lb[i]) / (nsteps - 1);

    I *= steps[i];
  }

  // Calculate the sum of the evaluations in the grid
  double cumulative = 0.;
  for (int j = 0; j < N; ++j) {

    double data[NDIM];

    int r = j;

    double fctr = p2;

    for (int k = 0; k < NDIM; ++k) {

      double rem = r % nsteps;

      data[k] = lb[k] + rem * steps[k];

      if (rem == 0 ||
          rem == (nsteps - 1)) // it is the first or the last element
        fctr /= 2;

      r /= nsteps;
    }

    cumulative += fctr * FUNCTION(DATA(data), FWD_PARAMS(params));
  }

  return I * cumulative;
}

// Code to evaluate a PDF on a binned data set numerically
#define EVALUATE_BINNED_NUMERICAL_CODE(i, out, ngaps, gaps_idx, gaps, edges,   \
                                       nsteps, params)                         \
  {                                                                            \
    double lower_bounds[NDIM];                                                 \
    double upper_bounds[NDIM];                                                 \
                                                                               \
    fill_bounds(lower_bounds, upper_bounds, i, ngaps, gaps_idx, gaps, edges);  \
                                                                               \
    int N = int_power(nsteps, NDIM);                                           \
                                                                               \
    out[i] = simpson_rule(N, nsteps, lower_bounds, upper_bounds, params);      \
  }

#ifdef USE_CPU
KERNEL void evaluate_binned_numerical(int len, double *out, int ngaps,
                                      int *gaps_idx, int *gaps, double *edges,
                                      int nsteps, double *params) {

  for (int i = 0; i < len; ++i) {
    EVALUATE_BINNED_NUMERICAL_CODE(i, out, ngaps, gaps_idx, gaps, edges, nsteps,
                                   params);
  }
}
#else
KERNEL void evaluate_binned_numerical(int lgth, GLOBAL_MEM double *out,
                                      int ngaps, GLOBAL_MEM int *gaps_idx,
                                      GLOBAL_MEM int *gaps,
                                      GLOBAL_MEM double *edges, int nsteps,
                                      GLOBAL_MEM double *params) {

  SIZE_T i = get_global_id(0);

  if (i >= lgth / NDIM)
    return;

  EVALUATE_BINNED_NUMERICAL_CODE(i, out, ngaps, gaps_idx, gaps, edges, nsteps,
                                 params);
}
#endif // USE_CPU

#ifdef INTEGRAL
// If the integral is defined, the evaluation in the binned case is also
// defined.
#define INTEGRAL_CODE(i, output_array, ngaps, gaps_idx, gaps, edges, params)   \
  {                                                                            \
                                                                               \
    double lower_bounds[NDIM];                                                 \
    double upper_bounds[NDIM];                                                 \
                                                                               \
    fill_bounds(lower_bounds, upper_bounds, i, ngaps, gaps_idx, gaps, edges);  \
                                                                               \
    output_array[i] =                                                          \
        INTEGRAL(DATA(lower_bounds), DATA(upper_bounds), FWD_PARAMS(params));  \
  }

#ifdef USE_CPU
void evaluate_binned(int len, double *output_array, int ngaps, int *gaps_idx,
                     int *gaps, double *edges, double *params) {

  for (int i = 0; i < len; ++i) {
    INTEGRAL_CODE(i, output_array, ngaps, gaps_idx, gaps, edges, params);
  }
}
#else
KERNEL void evaluate_binned(int lgth, GLOBAL_MEM double *output_array,
                            int ngaps, GLOBAL_MEM int *gaps_idx,
                            GLOBAL_MEM int *gaps, GLOBAL_MEM double *edges,
                            GLOBAL_MEM double *params) {

  SIZE_T i = get_global_id(0);

  if (i >= lgth)
    return;

  INTEGRAL_CODE(i, output_array, ngaps, gaps_idx, gaps, edges, params);
}
#endif // USE_CPU

#endif // INTEGRAL

#ifdef USE_CPU
}
#endif // USE_CPU
