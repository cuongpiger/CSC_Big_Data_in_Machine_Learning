import pyspark
import numpy as np
from pyspark import SparkContext
from time import time

from typing import List, Callable, List


def measureRun(function):
    start = time()
    function
    end = time()

    return '{} mili-seconds'.format(round((end - start)*1000, 7))


class CRDD:
    def __init__(self, rdd: pyspark.RDD):
        self.rdd = rdd

    def __len__(self):
        # return len(self.rdd.collect())
        return self.rdd.count()
    
    def getFirst(self) -> list:
        """ Lấy ra record đầu tiên từ `self.rdd`

        Returns:
            (list): record đầu tiên của `self.rdd`
        """
        return self.rdd.first()

    def getHead(self, amount: int = 5) -> List[str]:
        """ Lấy ra `amount` record đầu tiên của `self.rdd`

        Args:
            amount (int): số lượng record cần lấy

        Returns:
            (List[str]): [description]
        """
        return self.rdd.take(amount)
    
    def filter(self, def_: Callable):
        return CRDD(self.rdd.filter(def_))
    
    def map(self, def_: Callable):
        return CRDD(self.rdd.map(def_))
    
    def toNdarray(self):
        """ Chuyển đổi `self.rdd` thành ndarray

        Returns:
            (np.ndarray):
        """
        return np.array(self.rdd.collect())
    
    def getSamples(self, fraction: float, duplicate: bool = False, seed: int = 69) -> 'CRDD':
        """ Lấy mẫu từ một rdd nào đó

        Args:
            fraction (float): có giá trị từ [0, 1], tức cần lấy bao nhiêu phần trăm mẫu từ `self.rdd`
            duplicate (bool, optional): mẫu dc lấy có cần chứa các record trùng nhau hay ko
            seed (int, optional): tương tự `np.seed()`. Defaults to 69.

        Returns:
            [type]: [description]
        """
        return CRDD(self.rdd.sample(duplicate, fraction, seed))
    
    def subtract(self, other: 'CRDD'):
        """ Lấy ra phần (self.rdd - (self.rdd [intersection] other.rdd))

        Args:
            other (CRDD): một CRDD khác

        Returns:
            (CRDD): CRDD mới
        """
        return CRDD(self.rdd.subtract(other.rdd))
    
    def unique(self):
        """ Lấy ra các record duy nhất

        Returns:
            CRDD: 
        """
        return CRDD(self.rdd.distinct())
    
    def merge(self, other: 'CRDD', how: str = 'cross'):
        """ Hàm kết hợp hai rdd

        Args:
            other (CRDD): một rdd khác
            how (str, optional):
                * `cross`: lấy từng record của `self` kết hợp vs từng record của `other`

        Returns:
            CRDD:
        """
        if how == 'cross':
            return CRDD(self.rdd.cartesian(other.rdd))
        
    def getAmountPartitions(self) -> int:
        """ Lấy ra số lượng partition mà rdd này tạo ra

        Returns:
            (int): số lượng partition của self.rdd
        """
        return self.rdd.getNumPartitions()
    
    def save(self, path: str, how='text'):
        """ Dùng để lưu rdd ra các dạng thư mục chứa các file

        Args:
            path (str): đường dẫn đến thư mục cần lưu thư mục file
            how (str, optional):
                * `text`: lưu dưới dạng text file
        """
        try:
            if how == 'text':
                self.rdd.saveAsTextFile(path)
        
            print('Success.')
        except:
            print('Error!')

class CPySpark:
    def __init__(self, app_name: str = None):
        self.sc = SparkContext(master='local', appName=app_name)

    def rdd(self, data: np.ndarray = None, file: str = None, min_partitions: int = None, cache=False) -> CRDD(pyspark.RDD):
        res: pyspark.RDD = None

        if data is not None:
            res = self.sc.parallelize(data)

        if file is not None:
            res = self.sc.textFile(file, minPartitions=min_partitions)

        if cache:
            res = res.cache()

        return CRDD(res)
