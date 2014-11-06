PyPhot
=========

PyPhot is a simple Python translation of DAOPHOT-Type
photometry procedures from the IDL AstroLib (Landsman 1993),
including aperture and PSF-fitting algorithms, with a few modest additions to
increase functionality and ease of use.  For those members of the astronomical
community that prefer Python to IDL or do not have IDL licenses, we hope that these codes
will allow fast, easy, and reliable photometric measurements.  These procedures are
currently used in the Pan-STARRS supernova pipeline and the HST CLASH/CANDELS
supernova analysis, and will be updated and expanded as our analysis requires.

Authors: David O. Jones, Daniel Scolnic, Steven A. Rodney

Corresponding Author:
     David O. Jones
     Johns Hopkins University
     djones@pha.jhu.edu
     http://www.pha.jhu.edu/~djones/

To install, just append the PyPhot directory to your PYTHONPATH
environment variable.  Then use these routines in your python scripts
after importing them.  For example:

from PyPhot import pkfit

Additional documentation is provided in the individual routines.  Please
contact me with any issues, bugs, or suggestions.
