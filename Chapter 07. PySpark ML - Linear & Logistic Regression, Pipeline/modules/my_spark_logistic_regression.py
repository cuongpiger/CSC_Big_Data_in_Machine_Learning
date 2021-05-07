import pyspark
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
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
        handler = LogisticRegression(featuresCol='features', labelCol=self.target_feature, predictionCol='prediction')
        self.model = handler.fit(self.train_data)
        self.coefficients = self.model.coefficients
        
    def evaluateTestData(self):
        test_predict = self.model.transform(self.test_data)
        
        return test_predict.groupBy(self.target_feature, 'prediction').count().show()
    
    def evaluate(self):
        test_predict = self.model.transform(self.test_data)
        
        multi_evaluator = MulticlassClassificationEvaluator()
        weighted_precision = multi_evaluator.evaluate(test_predict, {multi_evaluator.metricName: 'weightedPrecision'})
        
        binary_evaluator = BinaryClassificationEvaluator()
        auc = binary_evaluator.evaluate(test_predict, {binary_evaluator.metricName: "areaUnderROC"})
        
        metrics = ['weighted precision', 'area under ROC']
        values = [weighted_precision, auc]
        
        return pd.DataFrame({
            'metric': metrics,
            'value': values
        })
        
    def saveModel(self, path: str):
        self.model.save(path)
        
def mySparkLogisticRegressionLoadModel(path: str):
    return LogisticRegressionModel.load(path)
        