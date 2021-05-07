import pyspark
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import LinearRegressionModel
from typing import List


class MySparkLinearRegression:
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
        self.train_data, self.test_data = self.df_data.randomSplit(
            (pTrainSize, 1 - pTrainSize))

    def describeTarget(self):
        train_target_col = self.target_feature + " (train)"
        test_target_col = self.target_feature + " (test)"

        train_describe: pyspark.sql.DataFrame = self.train_data.select(
            self.target_feature).describe().withColumnRenamed(self.target_feature, train_target_col)
        test_describe: pyspark.sql.DataFrame = self.test_data.select(
            self.target_feature).describe().withColumnRenamed(self.target_feature, test_target_col)
        entire_describe: pyspark.sql.DataFrame = train_describe.join(
            test_describe, on="summary")

        return entire_describe.show()

    def buildModel(self):
        handler = LinearRegression(featuresCol='features', labelCol=self.target_feature, predictionCol=(
            'predict_' + self.target_feature))
        self.model: pyspark.ml.regression.LinearRegressionModel = handler.fit(
            self.train_data)
        self.coefficients: np.ndarray = self.model.coefficients.toArray()
        self.intercept: float = self.model.intercept

    def evaluate(self):
        test_results = self.model.evaluate(self.test_data)
        metrics = ('MAE', 'MSE', 'RMSE', 'R-SQUARED')
        values = (
            test_results.meanAbsoluteError,
            test_results.meanSquaredError,
            test_results.rootMeanSquaredError,
            test_results.r2
        )
        evaluate_table = pd.DataFrame({
            'Metric': metrics,
            'Value': values
        })

        return evaluate_table, test_results.residuals

    def evaluateTestData(self, pVisual=False):
        test_predict = self.model.transform(self.test_data)

        if pVisual:
            def scatter():
                plt.figure(figsize=(8, 6))
                plt.scatter(test_predict.select(self.target_feature).collect(),
                            test_predict.select('predict_' + self.target_feature).collect())
                plt.xlabel("Actual values", color='b', weight='bold')
                plt.ylabel("Predictive values", color='b', weight='bold')
                plt.title("Compare actual and predictive values\n",
                          fontsize=18, color='r', weight='bold')

                plt.show()

            scatter()

        return test_predict

    def predict(self, pNewData):
        return self.model.predict(pNewData)
            

    def saveModel(self, path: str):
        self.model.save(path)
            

def mySparkLinearRegressionLoadModel(path: str) -> pyspark.ml.regression.LinearRegressionModel:
    return LinearRegressionModel.load(path)
