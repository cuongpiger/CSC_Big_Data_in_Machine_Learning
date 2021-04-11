import pyspark
import numpy as np
from pyspark import SparkContext

class CPySpark:
    def __init__(self, app_name: str = None):
        self.sc = SparkContext(master='local', appName=app_name)

    def readData(self, data_path: str):
        try:
            self.data = self.sc.textFile(data_path).cache()
        except:
            print("Error!")

    def rdd(self, data: np.ndarray = None, file: str = None, min_partitions: int = None):
        if data is not None:
            return self.sc.parallelize(data)

        if file is not None:
            return self.sc.textFile(file, minPartitions=min_partitions)