import pyspark
import pandas as pd

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from typing import List
from modules.my_spark_preprocessing import MyDataClassifier

class MyRandomForest:
    def __init__(self, pData: MyDataClassifier):
        self.data = pData
        self.model = None
        self.test_prediction = None
        
    def build(self):
        dtc = RandomForestClassifier(labelCol=self.data.output_var, featuresCol='features')
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
        

    