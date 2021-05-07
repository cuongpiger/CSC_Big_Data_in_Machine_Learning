import pyspark

from pyspark.ml.feature import StringIndexer, OneHotEncoder
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