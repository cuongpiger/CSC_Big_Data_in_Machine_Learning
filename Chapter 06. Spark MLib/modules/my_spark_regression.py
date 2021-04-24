from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import corr
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

import pyspark
from typing import List


class MySparkRegression:
    def __init__(self, data: pyspark.RDD):
        self.data = data
        self.final_data = None
        self.train = None
        self.test = None
        self.input: List[str] = None
        self.output: str = None
        self.model = None

    def transform(self, input: List[str], output: str, input_label: str = None):
        if input_label is None:
            input_label = 'features'
            
        self.input = input
        self.input_label = input_label
        self.output = output
        
        assembler = VectorAssembler(inputCols=self.input, outputCols=self.input_label)
        self.final_data = assembler.transform(self.data).select(self.input_label, self.output)
        
    def prepareData(self, train_size: float = .75):
        self.train, self.test = self.final_data.randomSplit((train_size, 1 - train_size))
        
    def buildModel(self):
        lr = LinearRegression(featuresCol=self.input_label, labelCol=self.output, predictionCol='predict_' + self.output)
        self.model = lr.fit(self.data)