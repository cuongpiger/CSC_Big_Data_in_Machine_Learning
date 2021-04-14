import pyspark
import numpy as np
from pyspark import SparkContext
from time import time


def measureRun(function):
    start = time()
    function
    end = time()

    return '{} mili-seconds'.format(round((end - start)*1000, 7))


class CPySpark:
    def __init__(self, app_name: str = None):
        self.sc = SparkContext(master='local', appName=app_name)

    def rdd(self, data: np.ndarray = None, file: str = None, min_partitions: int = None, cache=False) -> pyspark.RDD:
        res: pyspark.RDD = None

        if data is not None:
            res = self.sc.parallelize(data)

        if file is not None:
            res = self.sc.textFile(file, minPartitions=min_partitions)

        if cache:
            res = res.cache()

        return res
