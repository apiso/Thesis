import numpy

def history_read(filename):
    f = open(filename, 'r')
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    
    line = f.readline()
    linev = line.split()
    
    n = 0
    for line in f:
        n = n + 1
        
    f.close()
        
    f = open(filename, 'r')
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    f.readline()
        
    array = 0 * numpy.ndarray(shape = (n, len(linev)))
    
    for i in range(n):
        line = f.readline().split()
        
        for j in range(len(linev)):
            array[i, j] = float(line[j])
            
    return linev, array
        
    