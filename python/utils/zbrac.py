import sys

def zbrac(func, x1, x2, factor=1.6, ntry=50, tighten=True, args=(),
          full_output=0):
    """
    Based on Numerical Recipes' zbrac to bracket a root by outward expansion.
    This version adds an option to tighten the bracketing interval as the
    bracketing search proceeds.  Using this option gives a much slower growth
    of the bracketing interval.  With extra coding the original interval could
    be used for step-sizes while keeping track of the two best guesses.

    Parameters
    ----------
    func : callable f(x,*args)
        Objective function to minimize.
    x1, x2 : float
        Bracketing interval.
    factor : float
        Grow limit.
    ntry : int
        Maximum number of iterations to perform.
    tighten : boolean
        Tighten bracketing interval with each unsuccessful guess.
    args : tuple
        Additional arguments (if present), passed to `func`.    

    Returns
    -------
    x1, x2 : float
        Bracket.
    f1, f2 : float (only if full_output == True)
        Objective function values in bracket.
    success : boolean (only if full_output == True)
        Whether root was bracketed
    funcalls : int (only if full_output == True)
        Number of function evaluations made.
    """
            
    if factor < 1:
        print 'Warning low growth of bracket interval:', factor

    if x1 == x2:
        print 'you must provide two disctinct guesses to zbrac'
        sys.exit()

    def outs():
        if full_output:
            return (x1,x2), (f1,f2), success, t
        else:
            return (x1 , x2) , success
    
    f1 = func(x1,*args) #uses *-operator to unpack arguments from a list 
    f2 = func(x2,*args)
    success = True
    for t in range(ntry):        
        if (f1 * f2 < 0):
            out = outs() ; return out
        if (abs(f1) < abs(f2)):
            xtemp = x2
            if tighten:
                x2 , f2 = x1 , f1
            x1 = x1 + factor * ( x1 - xtemp )
            f1 = func(x1,*args)
        else:
            xtemp = x1
            if tighten:
                x1 , f1 = x2 , f2
            x2 = x2 + factor * ( x2 - xtemp )
            f2 = func(x2,*args)
        if (f1 * f2 == 0):
            print "Warning: root at edge of bracket interval"
            out = outs() ; return out    

    success = False
    out = outs() ; return out


