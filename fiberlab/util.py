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


def polygon_winding_number(polygon, point):
    """
    Determine the winding number of a 2D polygon about a point.
    
    The code does **not** check if the polygon is simple (no interesecting line
    segments).  Algorithm taken from Numerical Recipes Section 21.4.

    Args:
        polygon (`numpy.ndarray`_):
            An Nx2 array containing the x,y coordinates of a polygon.
            The points should be ordered either counter-clockwise or
            clockwise.
        point (`numpy.ndarray`_):
            One or more points for the winding number calculation.
            Must be either a 2-element array for a single (x,y) pair,
            or an Nx2 array with N (x,y) points.

    Returns:
        :obj:`int`, `numpy.ndarray`: The winding number of each point with
        respect to the provided polygon. Points inside the polygon have winding
        numbers of 1 or -1; see :func:`point_inside_polygon`.

    Raises:
        ValueError:
            Raised if ``polygon`` is not 2D, if ``polygon`` does not have two
            columns, or if the last axis of ``point`` does not have 2 and only 2
            elements.
    """
    # Check input shape is for 2D only
    if len(polygon.shape) != 2:
        raise ValueError('Polygon must be an Nx2 array.')
    if polygon.shape[1] != 2:
        raise ValueError('Polygon must be in two dimensions.')
    _point = numpy.atleast_2d(point)
    if _point.shape[1] != 2:
        raise ValueError('Point must contain two elements.')

    # Get the winding number
    nvert = polygon.shape[0]
    npnt = _point.shape[0]

    dl = numpy.roll(polygon, 1, axis=0)[None,:,:] - _point[:,None,:]
    dr = polygon[None,:,:] - point[:,None,:]
    dx = dl[...,0]*dr[...,1] - dl[...,1]*dr[...,0]

    indx_l = dl[...,1] > 0
    indx_r = dr[...,1] > 0

    wind = numpy.zeros((npnt, nvert), dtype=int)
    wind[indx_l & numpy.logical_not(indx_r) & (dx < 0)] = -1
    wind[numpy.logical_not(indx_l) & indx_r & (dx > 0)] = 1

    return numpy.sum(wind, axis=1)[0] if point.ndim == 1 else numpy.sum(wind, axis=1)


def point_inside_polygon(polygon, point):
    """
    Determine if one or more points is inside the provided polygon.

    Primarily a wrapper for :func:`polygon_winding_number`, that
    returns True for each point that is inside the polygon.

    Args:
        polygon (`numpy.ndarray`_):
            An Nx2 array containing the x,y coordinates of a polygon.
            The points should be ordered either counter-clockwise or
            clockwise.
        point (`numpy.ndarray`_):
            One or more points for the winding number calculation.
            Must be either a 2-element array for a single (x,y) pair,
            or an Nx2 array with N (x,y) points.

    Returns:
        :obj:`bool`, `numpy.ndarray`: Boolean indicating whether or not each
        point is within the polygon.
    """
    return numpy.absolute(polygon_winding_number(polygon, point)) == 1


