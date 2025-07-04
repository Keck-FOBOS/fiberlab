"""
.. include:: ../include/links.rst
"""
import numpy
from scipy import stats, optimize, signal, interpolate
from skimage import measure
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting

from IPython import embed


def sigma_clip_stdfunc_mad(data, **kwargs):
    """
    A simple wrapper for `scipy.stats.median_abs_deviation`_ that omits NaN
    values and rescales the output to match a normal distribution for use in
    `astropy.stats.sigma_clip`_.

    Args:
        data (`numpy.ndarray`_):
            Data to clip.
        **kwargs:
            Passed directly to `scipy.stats.median_abs_deviation`_.

    Returns:
        scalar-like, `numpy.ndarray`_: See `scipy.stats.median_abs_deviation`_.
    """
    return stats.median_abs_deviation(data, **kwargs, nan_policy='omit', scale='normal')


class Circle:
    """
    Fit the center and radius of a circle fit to a set of data.
    
    Args:
        x (array):
            The Cartesian x coordinates of the circle
        y (array):
            The Cartesian y coordinates of the circle
    """
    def __init__(self, x, y):
        self.x = numpy.atleast_1d(x)
        self.y = numpy.atleast_1d(y)

    def _fom(self, p):
        """
        Compute the figure-of-merit that the optimization method
        should minimize.
        """
        return numpy.sqrt((self.x-p[0])**2 + (self.y-p[1])**2) - p[2]

    def fit(self):
        """
        Determine the best-fitting center for the circle.

        Returns:
            array: Two-element array with the best-fitting center (x,y).
        """
        cx = numpy.median(self.x)
        cy = numpy.median(self.y)
        cr = numpy.median(numpy.sqrt((self.x-cx)**2 + (self.y - cy)**2))
        p0 = numpy.array([cx, cy, cr])
        lb = numpy.array([numpy.amin(self.x), numpy.amin(self.y), 0.])
        ub = numpy.array([numpy.amax(self.x), numpy.amax(self.y),
                          numpy.amax(numpy.sqrt(self.x**2+self.y**2))])

        r = optimize.least_squares(self._fom, p0, method='trf', bounds=(lb, ub),
                                   diff_step=numpy.full(p0.size, 1e-4, dtype=float)) #, verbose=2)

        return r.x

    @staticmethod
    def sample(p, n=1000):
        theta = numpy.linspace(-numpy.pi, numpy.pi, n)
        return p[2]*numpy.cos(theta) + p[0], p[2]*numpy.sin(theta) + p[1]

    @staticmethod
    def polar(x, y, p):
        _x = x - p[0]
        _y = y - p[1]
        return numpy.sqrt(_x**2 + _y**2), numpy.arctan2(_y, _x)
    
    @staticmethod
    def area(p):
        return numpy.pi * p[2]**2



class Ellipse:
    """
    Fit an ellipse to a set of coordinates.

    Equations are:
        
    Args:
        x (array):
            The Cartesian x coordinates of the polygon to model.
        y (array):
            The Cartesian y coordinates of the polygon to model.
    """
    def __init__(self, x, y):
        self.x = numpy.atleast_1d(x)
        self.y = numpy.atleast_1d(y)

    def _fom(self, p):
        """
        Compute the figure-of-merit that the optimization method
        should minimize.
        """
        cosa = numpy.cos(numpy.radians(p[2]))
        sina = numpy.sin(numpy.radians(p[2]))
        x = (self.x - p[0]) * cosa - (self.y - p[1]) * sina
        y = (self.x - p[0]) * sina + (self.y - p[1]) * cosa
        return numpy.absolute(numpy.sqrt(x**2 + (y/p[3])**2) - p[4])

    def fit(self):
        """
        Determine the best-fitting ellipse parameters
        """
        cx = numpy.median(self.x)
        cy = numpy.median(self.y)
        rot = 45.
        q = 0.5
        a = numpy.median(numpy.sqrt((self.x-cx)**2 + (self.y - cy)**2))
        p0 = numpy.array([cx, cy, rot, q, a])
        lb = numpy.array([numpy.amin(self.x), numpy.amin(self.y), 0., 0., 0.])
        ub = numpy.array([numpy.amax(self.x), numpy.amax(self.y), 180., 1., 
                          numpy.amax(numpy.sqrt(self.x**2+self.y**2))])

        r = optimize.least_squares(self._fom, p0, method='trf', bounds=(lb, ub),
                                   diff_step=numpy.full(p0.size, 1e-4, dtype=float), verbose=2)
        return r.x

    @staticmethod
    def sample(p, n=1000):
        theta = numpy.linspace(-numpy.pi, numpy.pi, n)
        _x = p[4] * numpy.cos(theta)
        _y = p[3] * p[4] * numpy.sin(theta)
        cosa = numpy.cos(numpy.radians(p[2]))
        sina = numpy.sin(numpy.radians(p[2]))
        return _x * cosa + _y * sina + p[0], - _x * sina + _y * cosa + p[1]
    
    @staticmethod
    def area(p):
        return numpy.pi * p[3] * p[4]**2

    @staticmethod
    def polar(x, y, p):
        cosa = numpy.cos(numpy.radians(p[2]))
        sina = numpy.sin(numpy.radians(p[2]))
        _x = (x - p[0]) * cosa - (y - p[1]) * sina
        _y = ((x - p[0]) * sina + (y - p[1]) * cosa)/p[3]
        return numpy.sqrt(_x**2 + _y**2), numpy.arctan2(_y, _x)


def get_bg(img, clip_iter=None, sigma_lower=100., sigma_upper=5.):
    """
    Measure the background in an image.

    Args:
        img (`numpy.ndarray`_, `numpy.ma.MaskedArray`_):
            2D array with image data.
        clip_iter (:obj:`int`, optional):
            Number of clipping iterations.  If None, no clipping is
            performed.
        sigma_lower (:obj:`float`, optional):
            Sigma level for clipping.  Clipping only removes negative outliers.
            Ignored if clip_iter is None.
        sigma_upper (:obj:`float`, optional):
            Sigma level for clipping.  Clipping only removes positive
            outliers.  Ignored if clip_iter is None.

    Returns:
        :obj:`tuple`: Returns the background level, the standard
        deviation in the background, and the number of rejected values
        excluded from the computation.
    """
    if clip_iter is None:
        # Assume the image has sufficient background pixels relative to
        # pixels with the fiber output image to find the background
        # using a simple median
        _img = img.compressed() if isinstance(img, numpy.ma.MaskedArray) else img
        bkg = numpy.median(_img)
        sig = stats.median_abs_deviation(_img, axis=None, nan_policy='omit', scale='normal')
        return bkg, sig, 0

    # Clip the high values of the image to get the background and
    # background error
    clipped_img = sigma_clip(img, sigma_lower=sigma_lower, sigma_upper=sigma_upper,
                             stdfunc=sigma_clip_stdfunc_mad, maxiters=clip_iter)
    bkg = numpy.ma.median(clipped_img)
    sig = stats.median_abs_deviation(clipped_img.compressed(), scale='normal')
    nrej = numpy.sum(numpy.ma.getmaskarray(clipped_img))

    return bkg, sig, nrej


def fit_bg(img, degree, sigma_lower=30., sigma_upper=3., maxclipiters=10, cenfunc='median',
           fititer=1):
    """
    """
    # Check the input
    if isinstance(degree, (int, numpy.integer)):
        _deg = (degree, degree)
    elif isinstance(degree, tuple):
        if len(degree) != 2:
            raise ValueError('Degree tuple must have two and only two elements.')
        _deg = degree
    else:
        raise TypeError('Degree must be an integer or a tuple of two integers')

    # Parse the image
    if isinstance(img, numpy.ma.MaskedArray):
        input_img = img.data
        input_bpm = numpy.ma.getmaskarray(img)
    else:
        input_img = numpy.atleast_2d(img)
        input_bpm = numpy.zeros(input_img.shape, dtype=bool)

    # Get the 2D coordinate arrays
    n = input_img.shape
    y, x = numpy.mgrid[:n[0],:n[1]]

    # Iteratively fit
    p_init = models.Legendre2D(_deg[0], _deg[1])
    fit_p = fitting.LinearLSQFitter()
    bg_model = numpy.zeros(n, dtype=float)
    for i in range(fititer):
        _img = numpy.ma.MaskedArray(input_img - bg_model, mask=input_bpm)
        _img = sigma_clip(_img, sigma_lower=sigma_lower, sigma_upper=sigma_upper,
                          maxiters=maxclipiters, cenfunc=cenfunc)
        gpm = numpy.logical_not(numpy.ma.getmaskarray(_img))
        print('fitting')
        p = fit_p(p_init, x[gpm], y[gpm], input_img[gpm])
        print('done')
        if i < fititer - 1:
            bg_model = p(x, y)

    return bg_model, x, y, gpm


def iterative_filter(data, window_length, polyorder, clip_iter=None, sigma=3., **kwargs):
    """
    Iteratively filter and reject 1D data.

    kwargs are passed directly to scipy.signal.savgol_filter.
    """
    if data.ndim > 1:
        raise ValueError('Data must be 1D.')
    if clip_iter is None:
        return signal.savgol_filter(data, window_length, polyorder, **kwargs)

    _sigma = numpy.squeeze(numpy.asarray([sigma]))
    if _sigma.size == 1:
        _sigma = numpy.array([sigma, sigma])
    elif _sigma.size > 2:
        raise ValueError('Either provide a single sigma for both lower and upper rejection, or '
                         'provide them separately.  Cannot provide more than two values.')

    nrej = 1
    i = 0
    _data = numpy.ma.MaskedArray(data)
    while i < clip_iter:
        bpm = numpy.ma.getmaskarray(_data)
        gpm = numpy.logical_not(bpm)
        _filt = signal.savgol_filter(_data[gpm], window_length, polyorder, **kwargs)
        tmp = sigma_clip(_data[gpm] - _filt, sigma_lower=_sigma[0], sigma_upper=_sigma[1],
                         stdfunc=sigma_clip_stdfunc_mad, maxiters=1)
        _bpm = numpy.ma.getmaskarray(tmp)
        if not numpy.any(_bpm):
            break
        bpm[gpm] |= _bpm
        _data[bpm] = numpy.ma.masked

    gpm = numpy.logical_not(bpm)
    _filt = signal.savgol_filter(data[gpm], window_length, polyorder, **kwargs)
    x = numpy.arange(data.size)
    return interpolate.interp1d(x[gpm], _filt)(x)


class ContourError(Exception):
    pass


# TODO:
#   - Allow to smooth the image before finding the contour
#   - Provide the contour level directly

def get_contour(img, level=None, threshold=None, bg=None, sig=None, clip_iter=10, sigma_lower=100.,
                sigma_upper=3.):
    """
    Return a single coherent contour of an image that contains the most number
    of contour points (as a proxy for the one that covers the most area).

    If ``sig`` and ``bg`` are None, ``img`` is used with :func:`get_bg` to get
    both (using the clipping arguments provided).  If ``sig`` is None, but
    ``bg`` is provided, it *must* be a `numpy.ndarray`_ and :func:`get_bg` is
    used to determine ``sig`` and a constant background.  If both ``sig`` and
    ``bg`` are provided, ``bg`` is subtracted directly from ``img``.

    Args:
        img (`numpy.ndarray`_):
            Image to contour
        level (float, optional):
            The exact level to contour.  If provided, all other keyword values
            are ignored.
        threshold (float, optional):
            The threshold in units of background sigma used to set the contour
            level.
        bg (float, `numpy.ndarray`_, optional):
            A background level to subtract.  It can be anything that broadcasts
            to the shape of ``img``.  
        sig (float, optional):
            The (assumed) constant noise level in the background of the image.  
        clip_iter (int, optional):
            Number of clipping iterations to use when measuring the background
            and noise level.  This is only used if sig or bg is None.  See
            :func:`get_bg`.
        sigma_lower (float, optional):
            Lower sigma rejection limit.  This is only used if sig or bg is
            None.  See :func:`get_bg`.
        sigma_upper (float, optional):
            Upper sigma rejection limit.  This is only used if sig or bg is
            None.  See :func:`get_bg`.

    Returns:
        tuple: The contour level, the contour coordinates, the noise level, and
        the background level.  The noise level will be None if the level is
        provided directly.
    """
    if level is None:
        # Compute the background flux, the standard deviation in the
        # background, and the number of rejected pixels
        if sig is None:
            if bg is None:
                bkg, sig, nrej = get_bg(img, clip_iter=clip_iter, sigma_lower=sigma_lower,
                                        sigma_upper=sigma_upper)
            else:
                bkg, sig, nrej = get_bg(bg, clip_iter=clip_iter, sigma_lower=sigma_lower,
                                        sigma_upper=sigma_upper)
        else:
            bkg = 0. if bg is None else bg

        # Find the contour matching the defined sigma threshold
        img_bksub = img - bkg
        if threshold is None:
            threshold = 2 * numpy.ma.std(img_bksub / sig)
        _level = threshold*sig
    else:
        bkg = 0. if bg is None else bg
        img_bksub = img - bkg
        _level = level
        sig = None

    # Transpose to match the numpy/matplotlib convention
    _img_bksub = img_bksub.filled(0.0) if isinstance(img_bksub, numpy.ma.MaskedArray) \
                    else img_bksub
    contour = measure.find_contours(_img_bksub.T, level=_level)
    if len(contour) == 0:
        raise ContourError('no contours found')

    # Only keep one contour with the most number of points.
    ci = -1
    nc = 0
    for i,p in enumerate(contour):
        if p.shape[0] > nc:
            nc = p.shape[0]
            ci = i

    return _level, contour[ci], sig, bkg


def growth_lim(a, lim, fac=1.0, midpoint=None, default=[0., 1.]):
    """
    Set the plots limits of an array based on two growth limits.

    Args:
        a (array-like):
            Array for which to determine limits.
        lim (:obj:`float`):
            Fraction of the total range of the array values to cover. Should
            be in the range [0, 1].
        fac (:obj:`float`, optional):
            Factor to contract/expand the range based on the growth limits.
            Default is no change.
        midpoint (:obj:`float`, optional):
            Force the midpoint of the range to be centered on this value. If
            None, set to the median of the data.
        default (:obj:`list`, optional):
            Default range to return if `a` has no data.

    Returns:
        :obj:`list`: Lower and upper limits for the range of a plot of the
        data in `a`.
    """
    # Get the values to plot
    _a = a.compressed() if isinstance(a, numpy.ma.MaskedArray) else numpy.asarray(a).ravel()
    if len(_a) == 0:
        # No data so return the default range
        return default

    # Sort the values
    srt = numpy.ma.argsort(_a)

    # Set the starting and ending values based on a fraction of the
    # growth
    _lim = 1.0 if lim > 1.0 else lim
    start = int(len(_a)*(1.0-_lim)/2)
    end = int(len(_a)*(_lim + (1.0-_lim)/2))
    if end == len(_a):
        end -= 1

    # Set the full range and increase it by the provided factor
    Da = (_a[srt[end]] - _a[srt[start]])*fac

    # Set the midpoint if not provided
    mid = (_a[srt[start]] + _a[srt[end]])/2 if midpoint is None else midpoint

    # Return the range for the plotted data
    return [ mid - Da/2, mid + Da/2 ]


def atleast_one_decade(lim):
    """
    Increase a provided set of limits so that they span at least one decade.

    Args:
        lim (array-like):
            A two-element object with, respectively, the lower and upper limits
            on a range.
    
    Returns:
        :obj:`list`: The adjusted lower and upper limits on the range.
    """
    lglim = numpy.log10(lim)
    if int(lglim[1]) - int(numpy.ceil(lglim[0])) > 0:
        return (10**lglim).tolist()
    m = numpy.sum(lglim)/2
    ld = lglim[0] - numpy.floor(lglim[0])
    fd = numpy.ceil(lglim[1]) - lglim[1]
    w = lglim[1] - m
    dw = ld*1.01 if ld < fd else fd*1.01
    _lglim = numpy.array([m - w - dw, m + w + dw])

    # TODO: The next few lines are a hack to avoid making the upper limit to
    # large. E.g., when lim = [ 74 146], the output is [11 1020]. This pushes
    # the middle of the range to lower values.
    dl = numpy.diff(_lglim)[0]
    if dl > 1 and dl > 3*numpy.diff(lglim)[0]:
        return atleast_one_decade([lim[0]/3,lim[1]])

    return atleast_one_decade((10**_lglim).tolist())
    

def rotate_y_ticks(ax, rotation, va):
    """
    Rotate all the existing y tick labels by the provided rotation angle
    (deg) and reset the vertical alignment.

    Args:
        ax (`matplotlib.axes.Axes`_):
            Rotate the tick labels for this Axes object. **The object is
            edited in place.**
        rotation (:obj:`float`):
            Rotation angle in degrees
        va (:obj:`str`):
            Vertical alignment for the tick labels.
    """
    for tick in ax.get_yticklabels():
        tick.set_rotation(rotation)
        tick.set_verticalalignment(va)


def convert_radius(r, pixelsize=None, distance=None, inverse=False):
    """
    Convert radius coordinates from pixels to mm or degrees.

    Args:
        r (`numpy.ndarray`_):
            Radius **in pixels** for the forward operation.  For the reverse
            operation (see ``inverse``), the radius should be in degrees if both
            ``pixelsize`` and ``distance`` are provided, or in mm if only
            ``pixelsize`` is provided.
        pixelsize (:obj:`float`, optional):
            Size of the image pixels in mm.
        distance (:obj:`float`, optional):
            Distance from the fiber output to the detector in mm.
        inverse (:obj:`bool`, optional):
            Perform the inverse operation; i.e., convert radius coordinates from
            mm or degrees to pixels.

    Returns:
        :obj:`tuple`: A string with the radius units and the updated radius
        values converted to either mm in the detector plane or output angle in
        degrees with respect to the fiber face normal vector.  For the reverse
        (see ``inverse``) operation, the returned units should be pixels.
    """
    _r = r.copy()
    if inverse:
        if pixelsize is not None and distance is not None :
            _r = distance * numpy.tan(numpy.radians(_r))
            r_units = 'mm'
        if pixelsize is not None :
            _r /= pixelsize 
            r_units = 'pix'
        return r_units, _r

    # Convert the radius from pixels to mm
    r_units = 'pix'
    if pixelsize is not None :
        _r *= pixelsize 
        r_units = 'mm'
    # Convert the radius from mm to angle
    if pixelsize is not None and distance is not None :
        _r = numpy.degrees(numpy.arctan(_r/distance))
        r_units = 'deg'
    return r_units, _r

