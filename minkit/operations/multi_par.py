'''
Functions depending on multiple arrays.
'''
import functools
from . import gpu_core

CACHE = {}

# This is the implementation of CUDA's atomicAdd in OpenCL
OPENCL_ATOMIC_ADD = '''
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

void atomicAdd(volatile __global double *addr, double val)
{
  union {
    long u;
    double f;
  } next, expected, current;
  current.f = *addr;
  do {
    expected.f = current.f;
    next.f = expected.f + val;
    current.u = atomic_cmpxchg( (volatile __global long *) addr,
				expected.u, next.u);
  } while( current.u != expected.u );
}
'''


def register(function):
    @functools.wraps(function)
    def __wrapper(n):
        if function.__name__ not in CACHE:
            CACHE[function.__name__] = {}
        if n not in CACHE[function.__name__]:
            CACHE[function.__name__][n] = function(n)
        return CACHE[function.__name__][n]
    return __wrapper


def code_for_bin_check(n):
    '''
    Generate the code to check whether a value is inside a bin.

    :param n: number of data dimensions.
    :type n: int
    :returns: code.
    :rtype: str
    '''
    idata = ','.join(f'GLOBAL_MEM double *data{i}' for i in range(n))
    igaps = ','.join(f'int gap{i}' for i in range(n))
    iedges = ','.join(f'GLOBAL_MEM double *edges{i}' for i in range(n))
    iborders = ','.join(f'GLOBAL_MEM unsigned *border{i}' for i in range(n))

    gap_sizes = '''
    int g0 = 1;
    '''
    for i in range(1, n):
        gap_sizes += f'''
        int g{i} = gap{i - 1} * g{i - 1};
        '''

    conds = ' && '.join(f'data{i}[idx] >= edges{i}[g{i} * idy] && '
                        f'(border{i} ? data{i}[idx] <= edges{i}[g{i} * (idy + 1)] '
                        f': data{i}[idx] < edges{i}[g{i} * (idy + 1)])' for i in range(n))

    code = f'''
    WITHIN_KERNEL unsigned bin_check( int idx, int idy, {idata}, {igaps}, {iedges}, {iborders} ) {{

       {gap_sizes}

       return ({conds});
    }}
    '''
    return code


def build_names(n, name, data_type):
    '''
    Build the strings to use for input/output of a variable.

    :param n: number of data dimensions.
    :type n: int
    :param name: name of the variable to use.
    :type name: str
    :param data_type: type of data (double, int, double*, int*, ...).
    :type data_type: str
    :returns: tuple with the names to be used for input and output.
    :rtype: tuple(str, str)
    '''
    t = tuple(f'{name}{i}' for i in range(n))
    o = ','.join(t)
    if '*' in data_type:
        i = ','.join(f'GLOBAL_MEM {data_type} {d}' for d in t)
    else:
        i = ','.join(f'{data_type} {d}' for d in t)
    return i, o


@register
def sum_inside_bins(n):
    '''
    Create a function to sum occurrences inside bins.

    :param n: number of data parameters.
    :type n: int
    :returns: compiled module.
    :rtype: function
    '''
    idata, odata = build_names(n, 'in', 'double*')
    igaps, ogaps = build_names(n, 'gap', 'int')
    iedges, oedges = build_names(n, 'edges', 'double*')
    iborders, oborders = build_names(n, 'border', 'unsigned*')

    preambulo = '' if gpu_core.BACKEND == gpu_core.CUDA else OPENCL_ATOMIC_ADD

    bin_check = code_for_bin_check(n)

    mod = gpu_core.THREAD.compile(f'''
    {preambulo}

    {bin_check}

    KERNEL void sum_inside_bins( GLOBAL_MEM double *out, {idata}, {igaps}, {iedges}, {iborders} )
    {{
       SIZE_T idx = get_global_id(0);
       SIZE_T idy = get_global_id(1);

       if ( bin_check(idx, idy, {odata}, {ogaps}, {oedges}, {oborders}) )
          atomicAdd(&out[idy], 1);
    }}
    ''')

    return mod.sum_inside_bins


@register
def sum_inside_bins_with_values(n):
    '''
    Create a function to sum occurrences inside bins.

    :param n: number of data parameters.
    :type n: int
    :returns: compiled module.
    :rtype: function
    '''
    idata, odata = build_names(n, 'in', 'double*')
    igaps, ogaps = build_names(n, 'gap', 'int')
    iedges, oedges = build_names(n, 'edges', 'double*')
    iborders, oborders = build_names(n, 'border', 'unsigned*')

    preambulo = '' if gpu_core.BACKEND == gpu_core.CUDA else OPENCL_ATOMIC_ADD

    bin_check = code_for_bin_check(n)

    mod = gpu_core.THREAD.compile(f'''
    {preambulo}

    {bin_check}

    KERNEL void sum_inside_bins_with_values( GLOBAL_MEM double *out, {idata}, {igaps}, {iedges}, {iborders}, GLOBAL_MEM double *values )
    {{
       SIZE_T idx = get_global_id(0);
       SIZE_T idy = get_global_id(1);

       if ( bin_check(idx, idy, {odata}, {ogaps}, {oedges}, {oborders}) )
          atomicAdd(&out[idy], values[idx]);
    }}
    ''')

    return mod.sum_inside_bins_with_values
