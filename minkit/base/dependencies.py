########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Functions to solve object dependencies.
'''

__all__ = []


def _solve_dependencies(objects):
    '''
    Get the resolution order for the given set of dependencies.

    :param objects: names of the objects with their dependencies.
    :type objects: list(tuple(str, list(str)))
    :returns: Resolution order.
    :rtype: list(str)
    '''
    ntot = len(objects)

    result = []

    objects = list(objects)
    while len(result) != ntot:
        new_objects = []
        for name, lst in objects:
            l = set(lst).difference(result)
            if len(l) == 0:
                result.append(name)
            else:
                new_objects.append((name, l))
        objects = new_objects
    return result


def _split_dependent_objects(obj):
    '''
    Split a registry depending whether they are *dependent* or not.

    :param obj: object to process.
    :type obj: PDF or ParameterBase
    :returns: Independent and dependent objects.
    '''
    iobjs = list(filter(lambda p: not p.dependent, obj.dependencies))
    dobjs = list(filter(lambda p: p.dependent, obj.dependencies))
    for p in dobjs:
        ip, dp = _split_dependent_objects(p)
        iobjs += ip
        dobjs += dp
    return iobjs, dobjs


def split_dependent_objects_with_resolution_order(objs):
    '''
    Split a set of objects in *independent* and *dependent* by recursively
    processing the possible dependencies.

    :param objs: collection of objects to process.
    :type objs: Registry
    :returns: Independent and dependent objects.
    '''
    iobjs, _dobjs = [], []
    for p in objs:
        if p.dependent:
            _dobjs.append(p)
            ip, dp = _split_dependent_objects(p)
            iobjs += [i for i in ip if not i in iobjs]
            _dobjs += [i for i in dp if not i in _dobjs]
        elif p not in iobjs:
            iobjs.append(p)

    ro = _solve_dependencies(
        [(d.name, {o.name for o in d.dependencies}.difference({o.name for o in iobjs})) for d in _dobjs])

    dobjs = []
    for n in ro:
        for o in _dobjs:
            if o.name == n and not o in dobjs:
                dobjs.append(o)
                break

    return iobjs, dobjs
