import pandas as pd
import pyspark
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import array, col, explode, lit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from modules.confusion_matrix import pretty_plot_confusion_matrix


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

def confusionMatrix(pDf: pyspark.sql.DataFrame):
    labels = [int(row[0] - 1) for row in pDf.select('label').collect()]
    counts = [int(row[0]) for row in pDf.select('count').collect()]
    predictions = [int(row[0] - 1) for row in pDf.select('prediction').collect()]


    n = len(set(labels + predictions))
    cm = np.zeros((n, n), dtype=int)
    
    for i in range(len(labels)):
        x = int(labels[i] - 1)
        y = int(predictions[i] - 1)
        cnt = int(counts[i])
        cm[x, y] = cnt
    
    df_cm = pd.DataFrame(cm, index=range(1, n + 1), columns=range(1, n + 1))
    cmap = 'PuRd'
    pretty_plot_confusion_matrix(df_cm, cmap=cmap)
    
def classifierMultiEvaluator(pDfPrediction: pyspark.sql.DataFrame, ):
    evaluator = MulticlassClassificationEvaluator()
    
    return pd.DataFrame({
        'Accuracy': [evaluator.evaluate(pDfPrediction, {evaluator.metricName: "accuracy"})],
        'F1-Score': [evaluator.evaluate(pDfPrediction, {evaluator.metricName: "f1"})],
        'Precision': [evaluator.evaluate(pDfPrediction, {evaluator.metricName: "weightedPrecision"})],
        'Recall': [evaluator.evaluate(pDfPrediction, {evaluator.metricName: "weightedRecall"})]
    }).T