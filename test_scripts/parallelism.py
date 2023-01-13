from multiprocessing import Pool

class parallel_engine(object):
    def __init__(self): pass
    def __call__(self, sk):
        pass # do something

if __name__ == '__main__':
    try:
        pool = Pool(2) # map 2 times
        engine = parallel_engine()
        pool.map(engine, sk8rs)
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()