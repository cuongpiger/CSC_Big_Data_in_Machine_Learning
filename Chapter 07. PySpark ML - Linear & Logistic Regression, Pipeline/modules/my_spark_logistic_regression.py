import pyspark
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from typing import List

class MySparkLogisticRegression:
    def __init__(self, pData: pyspark.RDD, pPredictorFeatures: List[str], pTargetFeature: str):
        self.data: pyspark.RDD = pData  # raw data
        self.predictor_features: List[str] = pPredictorFeatures
        self.target_feature: str = pTargetFeature
        self.df_data: pyspark.sql.DataFrame = None
        self.train_data: pyspark.sql.DataFrame = None
        self.test_data: pyspark.sql.DataFrame = None
        self.model = None
        self.coefficients = None
        self.intercept = None
        self.vector_assembler = None
        
    def prepareData(self):
        """ Phương thức này dùng để chuẩn bị dữ liệu cho model, cụ thể nó sẽ chuyển tất cả các thuộc tính
                trong `self.predictor_features` thành một vector với số chiều la length của 
                `self.predictor_features`, cuối cùng đặt tên cho vector này là `features`
        """
        self.vector_assembler = VectorAssembler(inputCols=self.predictor_features, outputCol='features')
        self.df_data = self.vector_assembler.transform(self.data).select('features', self.target_feature)
        
    def trainTestSplit(self, pTrainSize: float = .75):
        """ Tách dữ liệu ban đầu thành train data và test data với size của train data dc quy định bởi
                tham số `pTrainSize`
        Args:
            pTrainSize (float, optional): Kích thước của `self.train_data`
        """
        self.train_data, self.test_data = self.df_data.randomSplit((pTrainSize, 1 - pTrainSize))
        
    def buildModel(self):
        handler = LogisticRegression(featuresCol='features', labelCol=self.target_feature, predictionCol=('predict_' + self.target_feature))
        self.model = handler.fit(self.train_data)