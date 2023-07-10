"""
Miscellaneous package utilities.

.. include:: ../include/links.rst
"""
import numpy


def all_subclasses(cls):
    """
    Collect all the subclasses of the provided class.

    The search follows the inheritance to the highest-level class.  Intermediate
    base classes are included in the returned set, but not the base class itself.

    Thanks to:
    https://stackoverflow.com/questions/3862310/how-to-find-all-the-subclasses-of-a-class-given-its-name

    Args:
        cls (object):
            The base class

    Returns:
        :obj:`set`: The unique set of derived classes, including any
        intermediate base classes in the inheritance thread.
    """
    return set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in all_subclasses(c)])


def boxcar_average(arr, boxcar):
    """
    Boxcar average an array.

    Args:
        arr (`numpy.ndarray`_):
            Array to average.  Currently cannot be masked.
        boxcar (:obj:`int`, :obj:`tuple`):
            Integer number of pixels to average.  If a single integer,
            all axes are averaged with the same size box.  If a
            :obj:`tuple`, the integer is defined separately for each
            array axis; length of tuple must match the number of array
            dimensions.

    Returns:
        `numpy.ndarray`_: The averaged array.  If boxcar is a single
        integer, the returned array shape is::
            
            tuple([s//boxcar for s in arr.shape])

        A similar operation gives the shape when boxcar has elements
        defined for each array dimension.  If the input array is not an
        integer number of boxcar pixels along a given dimension, the
        remainder of the array elements along that dimension are ignored
        (i.e., pixels within the modulus of the array shape and boxcar
        of the end of the array dimension are ignored).
    """
    # Check and configure the input
    _boxcar = (boxcar,)*arr.ndim if isinstance(boxcar, int) else boxcar
    if not isinstance(_boxcar, tuple):
        raise TypeError('Input `boxcar` must be an integer or a tuple.')
    if len(_boxcar) != arr.ndim:
        raise ValueError('Must provide an integer or tuple with one number per array dimension.')

    # Perform the boxcar average over each axis and return the result
    _arr = arr.copy()
    for axis, box in zip(range(arr.ndim), _boxcar):
        _arr = numpy.add.reduceat(_arr, numpy.arange(0, _arr.shape[axis], box), axis=axis)/box
    return _arr


def boxcar_replicate(arr, boxcar):
    """
    Boxcar replicate an array.

    Args:
        arr (`numpy.ndarray`_):
            Array to replicate.
        boxcar (:obj:`int`, :obj:`tuple`):
            Integer number of times to replicate each pixel. If a
            single integer, all axes are replicated the same number
            of times. If a :obj:`tuple`, the integer is defined
            separately for each array axis; length of tuple must
            match the number of array dimensions.

    Returns:
        `numpy.ndarray`_: The block-replicated array.
    """
    # Check and configure the input
    _boxcar = (boxcar,)*arr.ndim if isinstance(boxcar, int) else boxcar
    if not isinstance(_boxcar, tuple):
        raise TypeError('Input `boxcar` must be an integer or a tuple.')
    if len(_boxcar) != arr.ndim:
        raise ValueError('Must provide an integer or tuple with one number per array dimension.')

    # Perform the boxcar average over each axis and return the result
    _arr = arr.copy()
    for axis, box in zip(range(arr.ndim), _boxcar):
        _arr = numpy.repeat(_arr, box, axis=axis)
    return _arr


