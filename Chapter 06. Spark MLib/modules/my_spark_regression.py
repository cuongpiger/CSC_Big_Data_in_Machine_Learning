import pyspark

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from typing import List


from modules.my_drawer import MyDrawer


class MySparkRegression:
    def __init__(self, pData: pyspark.RDD, pPredictorFeatures: List[str], pTargetFeature: str):
        self.data: pyspark.RDD = pData  # raw data
        # features from `pData` used to training model
        self.predictor_features: List[str] = pPredictorFeatures
        self.target_feature: str = pTargetFeature
        self.df_data: pyspark.sql.DataFrame = None
        self.train_data: pyspark.sql.DataFrame = None
        self.test_data: pyspark.sql.DataFrame = None

    def prepareData(self):
        """ Phương thức này dùng để chuẩn bị dữ liệu cho model, cụ thể nó sẽ chuyển tất cả các thuộc tính
                trong `self.predictor_features` thành một vector với số chiều la length của 
                `self.predictor_features`, cuối cùng đặt tên cho vector này là `features`
        """
        vector_assembler = VectorAssembler(
            inputCols=self.predictor_features, outputCol='features')
        self.df_data = vector_assembler.transform(
            self.data).select('features', self.target_feature)

    def trainTestSplit(self, pTrainSize: float = .75):
        """ Tách dữ liệu ban đầu thành train data và test data với size của train data dc quy định bởi
                tham số `pTrainSize`
        Args:
            pTrainSize (float, optional): Kích thước của `self.train_data`
        """
        if self.df_data is None:
            print("🚫 Please call method prepareData() before calling this method!")
            return

        self.train_data, self.test_data = self.df_data.randomSplit(
            (pTrainSize, 1 - pTrainSize))


    def describeTarget(self, visual=False):
        train_target_col = self.target_feature + " (train)"
        test_target_col = self.target_feature + " (test)"
        
        train_describe: pyspark.sql.DataFrame = self.train_data.select(self.target_feature).describe().withColumnRenamed(self.target_feature, train_target_col)
        test_describe: pyspark.sql.DataFrame = self.test_data.select(self.target_feature).describe().withColumnRenamed(self.target_feature, test_target_col)
        entire_describe: pyspark.sql.DataFrame = train_describe.join(test_describe, on="summary")
        
        if visual:
            describeTargetVisual()
        
        return train_describe.join(test_describe, on="summary")
    
    
    
def describeTargetVisual(pDesribeDataFrame: pyspark.sql.DataFrame):
    y_scale = pDesribeDataFrame.select('summary')