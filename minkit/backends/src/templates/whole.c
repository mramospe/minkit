/******************************************************************************
 * Template to define the file to be compiled. A backend greater than 0 means
 * that it will be comiled on CPU.
 ******************************************************************************/
#if {backend} > 0
#define USE_CPU
#else
#define USE_GPU
#endif

// Number of data dimensions that the funcion uses. Necessary to do the
// calculations
#define NDIM {ndimensions}
#define NPARS {number_of_parameters}

#if {has_variable_parameters}
#define HAS_VARIABLE_PARAMETERS
#endif

#ifdef USE_CPU
#include <cmath>
using namespace std;
#define WITHIN_KERNEL static inline
#define GLOBAL_MEM
#define KERNEL
#endif

{preamble_code}

{function}

{integral}

{evaluators}
