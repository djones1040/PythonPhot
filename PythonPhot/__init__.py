# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This is an Astropy affiliated package.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from . import aper
    from . import djs_angle_match
    from . import find
    from . import getpsf
    from . import pkfit
    from . import pkfit_noise
    from . import pkfit_norecenter
    from . import pkfit_norecent_noise
    from . import cntrd
    from . import hex2dec
    from . import meanclip
    from . import rebin
    from . import rdpsf
    from . import pixwt
    from . import mmm
    from . import daoerf
    from . import dao_value
    from . import photfunctions
    from . import pixwt
