import numpy
from scipy import stats, optimize, signal, interpolate
from skimage import measure
from astropy.stats import sigma_clip

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
        theta = numpy.linspace(-numpy.pi, numpy.pi, 1000)
        return p[2]*numpy.cos(theta) + p[0], p[2]*numpy.sin(theta) + p[1]

    @staticmethod
    def polar(x, y, p):
        _x = x - p[0]
        _y = y - p[1]
        return numpy.sqrt(_x**2 + _y**2), numpy.arctan2(_y, _x)



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
        theta = numpy.linspace(-numpy.pi, numpy.pi, 1000)
        _x = p[4] * numpy.cos(theta)
        _y = p[3] * p[4] * numpy.sin(theta)
        cosa = numpy.cos(numpy.radians(p[2]))
        sina = numpy.sin(numpy.radians(p[2]))
        return _x * cosa + _y * sina + p[0], - _x * sina + _y * cosa + p[1]

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
        img (array):
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
        bkg = numpy.median(img)
        sig = stats.median_abs_deviation(img, axis=None, nan_policy='omit', scale='normal')
        return bkg, sig, 0

    # Clip the high values of the image to get the background and
    # background error
    clipped_img = sigma_clip(img, sigma_lower=sigma_lower, sigma_upper=sigma_upper,
                             stdfunc=sigma_clip_stdfunc_mad, maxiters=clip_iter)
    bkg = numpy.ma.median(clipped_img)
    sig = stats.median_abs_deviation(clipped_img.compressed(), scale='normal')
    nrej = numpy.sum(numpy.ma.getmaskarray(clipped_img))

    return bkg, sig, nrej


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


# TODO: Allow to smooth the image before finding the contour
def get_contour(img, threshold=None, bg=None, sig=None, clip_iter=10, sigma_lower=100.,
                sigma_upper=3.):

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
        bkg = 0.

    # Find the contour matching the defined sigma threshold
    img_bksub = img - bkg
    if threshold is None:
        threshold = 2 * numpy.std(img_bksub / sig)
    level = threshold*sig

    # Transpose to match the numpy/matplotlib convention
    contour = measure.find_contours(img_bksub.T, level=level)
    if len(contour) == 0:
        raise ContourError('no contours found')

    # Only keep one contour with the most number of points.
    ci = -1
    nc = 0
    for i,p in enumerate(contour):
        if p.shape[0] > nc:
            nc = p.shape[0]
            ci = i

    return level, contour[ci], sig, bkg


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

