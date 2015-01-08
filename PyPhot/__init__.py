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
    import aper
    import djs_angle_match
    import find
    import getpsf
    import pkfit
    import pkfit_noise
    import pkfit_norecent
    import pkfit_norecent_noise
    import cntrd
    import hex2dec
    import meanclip
    import rebin
    import rdpsf
    import pixwt
    import mmm
    import daoerf
    import dao_value
