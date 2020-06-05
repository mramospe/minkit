########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Utils to bind class methods.
'''
import inspect
import functools

__all__ = []


def bind_method_arguments(method, **binded_kwargs):
    '''
    Bind the arguments of a method so they are replace by those given.

    :param method: method to bind.
    :type method: bound method
    :param binded_kwargs: keyword arguments to bind.
    :type binded_kwargs: dict
    :returns: wrapper function
    :rtype: function
    '''
    method_pars = inspect.signature(method).parameters  # ordered dictionary

    arg_names = list(p.name for p in filter(
        lambda p: p.default == inspect.Parameter.empty, method_pars.values()))

    # Get the argument names that must be specified
    available_args = []
    for i, n in enumerate(arg_names):
        if n not in binded_kwargs:
            available_args.append(n)
        else:
            break

    # Keep only those arguments that are needed for the function call
    replace_kwargs = {k: binded_kwargs[k] for k in filter(
        lambda s: s in method_pars, binded_kwargs)}

    @functools.wraps(method)
    def __wrapper(self, *args, **kwargs):
        '''
        Internal wrapper to execute "method" checking the input arguments.
        '''
        # Check that the call is done with the same arguments
        for name, v in kwargs.items():
            if name in replace_kwargs and replace_kwargs[name] is not v:
                raise ValueError(
                    f'Positional argument "{name}" is being called with a different input value')

        trueargs = []
        for name, arg in zip(method_pars, args):
            if name in replace_kwargs:
                if replace_kwargs[name] is not arg:
                    raise ValueError(
                        f'Keyword argument "{name}" is being called with a different input value')
            else:
                trueargs.append(arg)

        # Replace values
        kwargs.update(replace_kwargs)

        return method(*trueargs, **kwargs)

    return __wrapper


def bind_class_arguments(cls, **kwargs):
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
    name = f'Bind{cls.__class__.__name__}'
    parents = (object,)
    members = {name: bind_method_arguments(
        method, **kwargs) for name, method in class_members if name not in ('__init__',)}
    attributes = inspect.getmembers(
        cls, predicate=lambda a: not inspect.isroutine(a))
    members.update(
        {name: a for name, a in attributes if not name.startswith('__')})
    members['__enter__'] = lambda self, *args, **kwargs: self
    members['__exit__'] = lambda self, *args, **kwargs: self
    BindObject = type(name, parents, members)
    return BindObject()
