import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import array, col, explode, lit


def createSparkDfFromXlsx(pDf: pd.DataFrame, pSpark: SparkSession):
    columns = pDf.columns.tolist()
    data = list(pDf.itertuples(index=False, name=None))
    rdd = pSpark.sparkContext.parallelize(data)
    
    return rdd.toDF(columns)
    
def oversampling(pDf: pyspark.sql.DataFrame, pColumn: str, pMajorValue, pMinorValue):
    major_df = pDf.filter(col(pColumn) == pMajorValue)
    minor_df = pDf.filter(col(pColumn) == pMinorValue)
    ratio = int(major_df.count() / minor_df.count())
    
    oversampled_df = minor_df.withColumn('dummy', explode(array([lit(x) for x in range(ratio)]))).drop('dummy')
    combined_df = major_df.unionAll(oversampled_df)
    
    return combined_df