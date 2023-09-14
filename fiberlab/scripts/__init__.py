
from ..util import all_subclasses
from . import scriptbase

from . import browse_images
from . import collimated_farfield
from . import collimated_FRD
from . import farfield_inspector
from . import fullcone_farfield
from . import fullcone_ee_fratio
from . import fullcone_throughput

# Build the list of script classes
def script_classes():
    import numpy

    # Recursively collect all subclasses
    scr_c = numpy.array(list(all_subclasses(scriptbase.ScriptBase)))
    scr_n = numpy.array([c.name() for c in scr_c])
    # Construct a dictionary with the script name and class
    srt = numpy.argsort(scr_n)
    return dict([ (n,c) for n,c in zip(scr_n[srt],scr_c[srt])])

fobos_fiberlab_scripts = list(script_classes().keys())



