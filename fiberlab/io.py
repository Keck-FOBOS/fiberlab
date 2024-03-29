"""
General input/output methods.

.. include:: ../include/links.rst
""" 

from pathlib import Path
import warnings
from IPython import embed
import numpy
from skimage import io
from astropy.io import fits


def bench_image(ifile, ext=0, compress=True):
    """
    Read an image from a bench imaging camera.

    Args:
        ifile (:obj:`str`, `Path`_):
            File name.  Can be None, but, if so, None is returned.
        ext (:obj:`int`, optional):
            If the file is a multi-extension fits file, this selects the
            extension with the relevant data.
        compress (:obj:`bool`, optional):
            If the data include multiple color channels (as indicated by the
            image data array being 3D), compress into a single 2D image by
            summing across the last dimension?

    Returns:
        `numpy.ndarray`_: Array with the (floating-point) image data.  If
        ``ifile`` is None, this is also None.
    """
    # No file so return None.
    if ifile is None:
        return None

    # Set the file path and check that it exists.
    _ifile = Path(ifile).resolve()
    if not _ifile.exists():
        raise FileNotFoundError(f'{_ifile} does not exist!')

    # Read fits files using astropy
    if any([s in ['.fit', '.fits'] for s in _ifile.suffixes]):
        return fits.open(_ifile)[ext].data.astype(float)

    # Read all other files using scikit-image
    img = io.imread(_ifile).astype(float)
    if compress and img.ndim == 3:
        img = numpy.sum(img, axis=2)

    return img


def gather_collimated_file_list(root, par=None, threshold=None):
    """
    Gather the list of collimated farfield output files to analyze.

    Args:
        root (:obj:`str`, `Path`_):
            Root directory with files.
        par (:obj:`str`, `Path`_, optional):
            Name of a file that provides 2 or 3 columns: (1) the files to
            analyze, (2) the background image to use for each file, and (3) the
            threshold to use for each file.  If this file is provided, any
            threshold is ignored and the root directory is not trolled for all
            bg*, z*, and a* files.  The last column with the thresholds can be
            omitted, which means the code will use the value provided on the
            command line (or its default).
        threshold (:obj:`float`, optional):
            If ``par`` is not provided or it has no 3nd column, this is the
            threshold to use for all files.

    Returns:
        :obj:`tuple`: Provides (1) the z0 file, (2) its background file, (3) its
        threshold, (4) the z1 file, (5) its background file, (6) its threshold,
        (7) the list of a* files, (8) their background files, and (9) their
        thresholds.
    """
    if par is None:
        z0, z1, bg, afiles = find_collimated_farfield_files(root)
        na = len(afiles)
        return z0, bg, threshold, z1, bg, threshold, afiles, [bg]*na, [threshold]*na

    return parse_file_list(root, par, threshold)


def gather_fullcone_file_list(root, par=None, threshold=None):
    """
    Gather the list of full-cone farfield output files to analyze.

    Args:
        root (:obj:`str`, `Path`_):
            Root directory with files.
        par (:obj:`str`, `Path`_, optional):
            Name of a file that provides 2 or 3 columns: (1) the files to
            analyze, (2) the background image to use for each file, and (3) the
            threshold to use for each file.  If this file is provided, any
            threshold is ignored and the root directory is not trolled for all
            bg*, z*, and a* files.  The last column with the thresholds can be
            omitted, which means the code will use the value provided on the
            command line (or its default).
        threshold (:obj:`float`, optional):
            If ``par`` is not provided or it has no 3nd column, this is the
            threshold to use for all files.

    Returns:
        :obj:`tuple`: Provides (1) the z0 file, (2) its background file, (3) its
        threshold, (4) the z1 file, (5) its background file, (6) its threshold,
        (7) the list of a* files, (8) their background files, and (9) their
        thresholds.
    """
    if par is None:
        z0, z1, bg = find_fullcone_farfield_files(root)
        return z0, bg, threshold, z1, bg, threshold
    return parse_file_list(root, par, threshold)[:6]


def find_collimated_farfield_files(root):
    """
    Find the set of collimated far-field output files according to the expected
    nomenclature.

    Args:
        root (:obj:`str`, `Path`_):
            Root directory with files.

    Returns:
        :obj:`tuple`: Full paths to (1) the z0 image, (2) the z1 image, (3) the
        background image, and (4) the list of a* files.
    """
    # Set the root directory
    _root = Path(root).resolve()
    if not _root.exists():
        raise FileNotFoundError(f'{_root} is not a valid directory.')

    # Find the z0 and z1 images
    z0 = sorted(list(_root.glob('z0*')))
    if len(z0) > 1:
        raise ValueError('More than one z0 image found.  Directory should only have one of '
                            f'the following: {[f.name for f in z0]}')
    if len(z0) == 0:
        raise ValueError('Could not find z0 image.')
    z0 = z0[0]
    z1 = sorted(list(_root.glob('z1*')))
    if len(z1) > 1:
        raise ValueError('More than one z1 image found.  Directory should only have one of '
                            f'the following: {[f.name for f in z1]}')
    if len(z1) == 0:
        raise ValueError('Could not find z1 image.')
    z1 = z1[0]

    # Attempt to find a background image
    bg = sorted(list(_root.glob('bg*')))
    if len(bg) >= 1:
        if len(bg) > 1:
            warnings.warn(f'Found more than one background image: {[f.name for f in bg]}.  '
                          'Using the first one.')
        bg = bg[0]
    else:
        warnings.warn('No background images found.  Will attempt to use main image to set '
                        'background level.')
        bg = None

    afiles = sorted(list(_root.glob('a*')))

    return z0, z1, bg, afiles


def find_fullcone_farfield_files(root):
    """
    Find a set of fullcone far-field output files according to the expected
    nomenclature.

    Args:
        root (:obj:`str`, `Path`_):
            Root directory with files.

    Returns:
        :obj:`tuple`: Full paths to (1) the z0 image, (2) the z1 image, and (3)
        the background image.
    """
    # Set the root directory
    _root = Path(root).resolve()
    if not _root.exists():
        raise FileNotFoundError(f'{_root} is not a valid directory.')

    # Find the z0 and z1 images
    z0 = sorted(list(_root.glob('z0*')))
    if len(z0) > 1:
        raise ValueError('More than one z0 image found.  Directory should only have one of '
                            f'the following: {[f.name for f in z0]}')
    if len(z0) == 0:
        raise ValueError('Could not find z0 image.')
    z0 = z0[0]
    z1 = sorted(list(_root.glob('z1*')))
    if len(z1) > 1:
        raise ValueError('More than one z1 image found.  Directory should only have one of '
                            f'the following: {[f.name for f in z1]}')
    if len(z1) == 0:
        raise ValueError('Could not find z1 image.')
    z1 = z1[0]

    # Attempt to find a background image
    bg = sorted(list(_root.glob('bg*')))
    for i in range(len(bg)):
        if bg[i].is_dir():
            bg[i] = sorted(list(bg[i].glob('bg*')))
    bg = numpy.concatenate(bg).tolist()
    if len(bg) >= 1:
        if len(bg) > 1:
            warnings.warn(f'Found more than one background image: {[f.name for f in bg]}.  '
                          'Using the first one.')
        bg = bg[0]
    else:
        warnings.warn('No background images found.  Will attempt to use main image to set '
                        'background level.')
        bg = None

    return z0, z1, bg


def parse_file_list(root, par, threshold):
    """
    Parse a file with the set of data files to analyze.

    Args:
        root (:obj:`str`, `Path`_):
            Root directory with files.
        par (:obj:`str`, `Path`_):
            Name of a file that provides 2 or 3 columns: (1) the files to
            analyze, (2) the background image to use for each file, and (3) the
            threshold to use for each file.  The last column with the thresholds
            can be omitted, which means the code will use the value provided on
            the command line (or its default).
        threshold (:obj:`float`, optional):
            If ``par`` has no 3nd column, this is the threshold to use for all
            files.

    Returns:
        :obj:`tuple`: Provides (1) the z0 file, (2) its background file, (3) its
        threshold, (4) the z1 file, (5) its background file, (6) its threshold,
        (7) the list of a* files, (8) their background files, and (9) their
        thresholds.
    """

    # Set the root directory
    _root = Path(root).resolve()
    if not _root.exists():
        raise FileNotFoundError(f'{_root} is not a valid directory.')

    _par = Path(par).resolve()
    if not _par.exists():
        raise FileNotFoundError(f'{_par} does not exist!')

    # Use numpy to read the file
    db = numpy.genfromtxt(str(_par), dtype=str)
    nfiles = db.shape[0]
    if db.shape[1] == 2:
        imgs, bkgs = db.T
        thresh = [threshold]*nfiles
        pseudo = [None] * nfiles
    elif db.shape[1] == 3:
        imgs, bkgs, thresh = db.T
        thresh = thresh.astype(float)
        pseudo = [None] * nfiles
    elif db.shape[1] == 4:
        imgs, bkgs, thresh, pseudo = db.T
        thresh = thresh.astype(float)
    else:
        raise ValueError(f'{_par} must only contain 2, 3, or 4 columns!')

    # Check the files exist in the root directory
    for i in range(nfiles):
        if not (_root / imgs[i]).exists():
            raise FileNotFoundError(f'{imgs[i]} is not a file in {_root}.')
        if not (_root / bkgs[i]).exists():
            raise FileNotFoundError(f'{bkgs[i]} is not a file in {_root}.')

    if nfiles == 2:
        return _root / imgs[0], _root / bkgs[0], thresh[0], \
                 _root / imgs[1], _root / bkgs[1], thresh[1], None, None, None

    # Find the z0 file
    indx = numpy.where([i[:2] == 'z0' or p == 'z0' for i,p in zip(imgs, pseudo)])[0]
    if len(indx) != 1:
        raise ValueError('There should one and only one file named/marked as "z0".')
    z0 = _root / imgs[indx[0]]
    z0_bg = _root / bkgs[indx[0]]
    z0_thresh = thresh[indx[0]]

    # Find the z1 file
    indx = numpy.where([i[:2] == 'z1' or p == 'z1' for i,p in zip(imgs, pseudo)])[0]
    if len(indx) != 1:
        raise ValueError('There should one and only one file named/marked as "z1".')
    z1 = _root / imgs[indx[0]]
    z1_bg = _root / bkgs[indx[0]]
    z1_thresh = thresh[indx[0]]

    # Find the angle-sweep files
    indx = numpy.where([i[0] == 'a' or (p is not None and p[0] == 'a')
                        for i,p in zip(imgs, pseudo)])[0]
    if nfiles != indx.size + 2:
        warnings.warn(f'Could not parse {nfiles - 2 - indx.size} files in {_par}.')
    if indx.size == 0:
        warnings.warn(f'Could not find any angle-sweep files.')

    return z0, z0_bg, z0_thresh, z1, z1_bg, z1_thresh, \
                _root / imgs[indx], _root / bkgs[indx], thresh[indx]


