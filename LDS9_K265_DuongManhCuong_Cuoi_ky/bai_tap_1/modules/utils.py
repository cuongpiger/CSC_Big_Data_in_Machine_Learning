import pandas as pd
from pyspark.sql import SparkSession


def createSparkDfFromXlsx(pDf: pd.DataFrame, pSpark: SparkSession):
    columns = pDf.columns.tolist()
    data = list(pDf.itertuples(index=False, name=None))
    rdd = pSpark.sparkContext.parallelize(data)
    
    return rdd.toDF(columns)
    