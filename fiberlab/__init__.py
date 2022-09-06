"""
fobos-producer package initialization.
"""

import os
import warnings

from .version import version

def short_warning(message, category, filename, lineno, file=None, line=None):
    """
    Return the format for a short warning message.
    """
    return ' %s: %s (%s:%s)\n' % (category.__name__, message, os.path.basename(filename), lineno)

warnings.formatwarning = short_warning

# Set version
__version__ = version

# Report current coverage
__coverage__ = 0.55


from pathlib import Path
from pkg_resources import resource_filename


def data_file(filename=None):
    root = Path(resource_filename('fiberlab', 'data')).resolve()
    return root if filename is None else root / filename


