import contextlib
import inspect
import functools


class dummy(object):

    def __init__( self ):
        super(dummy, self).__init__()

    def cache( self, positional, positional2, values = None, parameters = None ):
        print(positional, positional2, values, parameters)

    def another( self, values = None ):
        print(values)

d = dummy()


def bind_method_arguments( method, **binded_kwargs ):
    '''
    Bind the arguments of a method so they are replace by those given.

    :param method: method to bind.
    :type method: bound method
    :param binded_kwargs: keyword arguments to bind.
    :type binded_kwargs: dict
    :returns: wrapper function
    :rtype: function
    '''
    argspec = inspect.getargspec(method)

    if argspec.defaults:
        arg_names   = argspec.args[1:-len(argspec.defaults)]
        kwarg_names = argspec.args[-len(argspec.defaults):]
    else:
        arg_names   = argspec.args[1:]
        kwarg_names = []

    # Check that the arguments are consumed in the right order
    args = list(map(lambda a: a in binded_kwargs, arg_names))
    if not all(f <= s for f, s in zip(args[:-1], args[1:])):
        raise RuntimeError(f'Problems parsing the positional arguments of "{method.__name__}"; make sure they are specified maintaining the order')

    # Dictionary to store the binded values
    replace_kwargs  = {b: binded_kwargs[b] for b in filter(lambda b: b in argspec.args, binded_kwargs)}

    # Store the names that are still available
    available_args   = list(filter(lambda a: a not in binded_kwargs, arg_names))
    available_kwargs = list(filter(lambda a: a not in binded_kwargs, kwarg_names))

    @functools.wraps(method)
    def __wrapper( self, *args, **kwargs ):
        '''
        Internal wrapper to execute "method" checking the input arguments.
        '''
        # Check the specified arguments
        if len(args) > len(arg_names):
            raise ValueError(f'Number of input arguments is greater than the function admits: args={available_args}, kwargs={available_kwargs}')

        # Check that the call is done with the same arguments
        for name, v in kwargs.items():
            if name in replace_kwargs and replace_kwargs[name] is not v:
                raise ValueError(f'Positional argument "{name}" is being called with a different input value')

        for name, arg in zip(arg_names, args):
            if name in replace_kwargs and replace_kwargs[name] is not arg:
                raise ValueError(f'Keyword argument "{name}" is being called with a different input value')

        # Replace values
        kwargs.update(replace_kwargs)

        for a in filter(lambda a: a in kwargs, arg_names[:len(args)]):
            kwargs.pop(a)

        return method(*args, **kwargs)

    return __wrapper


def bind_class_arguments( cls, **kwargs ):
    '''
    Dinamically create a new class based on "cls", where all the methods are wrapped,
    so the input arguments are replaced (if they exist) by those in "kwargs".
    The resulting class can be used as a context manager.

    :param cls: class to wrap.
    :type cls: class
    :param kwargs: arguments to replace.
    :type kwargs: dict
    :returns: wraper around the base class.
    :rtype: class
    '''
    class_members = inspect.getmembers(cls, predicate=inspect.ismethod)
    name    = f'Bind{cls.__class__.__name__}'
    parents = (object,)
    members = {name: bind_method_arguments(method, **kwargs) for name, method in class_members}
    members['__enter__'] = lambda self, *args, **kwargs: self
    members['__exit__'] = lambda self, *args, **kwargs: self
    BindObject = type(name, parents, members)
    return BindObject()


with bind_class_arguments(dummy(), positional2=1, values = [1, 2]) as obj:
    obj.cache(1, 1, parameters=2)
    obj.another()
