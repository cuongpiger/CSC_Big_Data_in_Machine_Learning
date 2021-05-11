import pyspark

from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from typing import List

def indexerData(pData: pyspark.sql.DataFrame, pInputCols: List[str]):
    outputCols = ["idx_" + name for name in pInputCols]
    indexer = StringIndexer(inputCols=pInputCols, outputCols=outputCols)
    
    return indexer.fit(pData).transform(pData)

def oneHotEncoder(pData: pyspark.sql.DataFrame, pInputCols: List[str]):
    data_indexer = indexerData(pData, pInputCols)
    inputCols = ['idx_' + name for name in pInputCols]
    outputCols = ['oh_' + name for name in pInputCols]
    encoder = OneHotEncoder(inputCols=inputCols, outputCols=outputCols, dropLast=False)
    
    return encoder.fit(data_indexer).transform(data_indexer)

def dropNa(pData: pyspark.sql.DataFrame):
    return pData.na.drop()


class MyDataClassifier:
    def __init__(self):
        self.raw_data = None
        self.input_vars = None
        self.output_var = None
        self.assembler = None
        self.data = None
        self.train_data = None
        self.test_data = None
    
    def initModel(self, pData: pyspark.RDD, pPredictorVariables: List[str], pTargetVariable: str):
        self.raw_data = pData
        self.input_vars = pPredictorVariables
        self.output_var = pTargetVariable
        self.assembler = VectorAssembler(inputCols=self.input_vars, outputCol='features')
        
    def prepareData(self):
        self.data = self.assembler.transform(self.raw_data).select('features', self.output_var)
        
    def trainTestSplit(self, pTrainSize: float = .75):
        if pTrainSize == 1:
            self.train_data = self.data
        else:
            self.train_data, self.test_data = self.data.randomSplit((pTrainSize, 1 - pTrainSize))
