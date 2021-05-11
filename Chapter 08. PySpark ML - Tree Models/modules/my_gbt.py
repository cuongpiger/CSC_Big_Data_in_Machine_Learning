import pyspark
import pandas as pd

from pyspark.ml.classification import GBTClassifier
from typing import List
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from modules.my_spark_preprocessing import MyDataClassifier

class MyGBT:
    def __init__(self, pData: MyDataClassifier):
        self.data = pData
        self.model = None
        self.test_prediction = None
        
    def build(self):
        dtc = GBTClassifier(labelCol=self.data.output_var, featuresCol='features')
        self.model = dtc.fit(self.data.train_data)
        
    def predictTestData(self):
        if self.data.test_data is not None:
            self.test_prediction = self.model.transform(self.data.test_data)
        
        return self.test_prediction
        
    def evaluate(self):
        evaluator = MulticlassClassificationEvaluator(labelCol=self.data.output_var, predictionCol='prediction', metricName='accuracy')
        
        metrics = ['Accuracy']
        values = [
            evaluator.evaluate(self.test_prediction)
        ]
        
        return pd.DataFrame({
            'Metric': metrics,
            'Values': values,
        })

    