#!/usr/bin/env python
#D. Jones - 1/14/14

def hex2dec(inp,verbose=True):
    """This code has been converted from the IDL source code
    at http://www.astro.washington.edu/docs/idl/cgi-bin/getpro/library32.html?HEX2DEC
    Convert hexadecimal representation to decimal integer.
               
    Explanation : A hexadecimal string is converted to a decimal integer and 
                  can be displayed or returned or both or neither.

    Use         : decimal = hex2dec.hex2dec(hex, verbose=True)
    Inputs      : hex - hexadecimal string
    Opt. Inputs : None
    Outputs     : See below
    Opt. Outputs: decimal - the decimal integer equivalent of the input.               
    Keywords    : verbose - if given the decimal number is printed to the
                             terminal.  Default = True.
    Calls       : None
    Restrictions: Input must be a string.
    Side effects: None
    Category    : Utils, Numerical
    Prev. Hist. : None
    Written     : C D Pike, RAL, 7-Oct-93
    Modified    : Converted to Python, D. Jones, January 2014
    Version     : Version 1, 7-Oct-93
"""

    #
    #  trap invalid input
    #
    
    #  
    #  initialise output etc
    #
    out = 0
    n = len(inp)
    
    #
    #  convert each character in turn
    #
    for i in range(n)[::-1]:
        try: c = (inp[i]).upper()
        except: c = inp[i]

        if c == 'A': c = 10
        elif c == 'B': c = 11
        elif c == 'C': c = 12
        elif c == 'D': c = 13
        elif c == 'E': c = 14
        elif c == 'F': c = 15
        else: 
            if c != int(c):
                print('Invalid character **',c,'**')
                out = 0
                return(out)
        
  
        out = out + int(c)*16**int(n-1-i)

#
#  if not silenced, print result
#
    if verbose: print(out)
    return(out)
