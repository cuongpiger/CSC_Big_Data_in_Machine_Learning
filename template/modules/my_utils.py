from time import time

def measureRun(function):
    start = time()
    function
    end = time()

    return '{} mili-seconds'.format((end - start)*1000)