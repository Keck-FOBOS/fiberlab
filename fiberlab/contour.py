
import numpy
from scipy import stats, optimize
from skimage import measure
from astropy.stats import sigma_clip


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



def get_bg(img, clip_iter=None, sigma_upper=5.):
    """
    Measure the background in an image.

    Args:
        img (array):
            2D array with image data.
        clip_iter (:obj:`int`, optional):
            Number of clipping iterations.  If None, no clipping is
            performed.
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
    clipped_img = sigma_clip(img, sigma_lower=100, sigma_upper=sigma_upper,
                             stdfunc=sigma_clip_stdfunc_mad, maxiters=clip_iter)
    bkg = numpy.ma.median(clipped_img)
    sig = stats.median_abs_deviation(clipped_img.compressed(), scale='normal')
    nrej = numpy.sum(numpy.ma.getmaskarray(clipped_img))

    return bkg, sig, nrej


def get_contour(img, threshold, bg=None, sig=None, clip_iter=10, sigma_upper=3.):

    # Compute the background flux, the standard deviation in the
    # background, and the number of rejected pixels
    if sig is None:
        if bg is None:
            bkg, sig, nrej = get_bg(img, clip_iter=clip_iter, sigma_upper=sigma_upper)
        else:
            bkg, sig, nrej = get_bg(bg, clip_iter=clip_iter, sigma_upper=sigma_upper)
    else:
        bkg = 0.

    # Find the contour matching the defined sigma threshold
    img_bksub = img - bkg
    level = threshold*sig

    # Transpose to match the numpy/matplotlib convention
    contour = measure.find_contours(img_bksub.T, level=level) 
    # Only keep one contour with the most number of points.
    ci = -1
    nc = 0
    for i,p in enumerate(contour):
        if p.shape[0] > nc:
            nc = p.shape[0]
            ci = i
    return contour[ci]


