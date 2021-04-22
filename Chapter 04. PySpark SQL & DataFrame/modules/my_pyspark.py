import pyspark
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.context import SQLContext

from typing import List


class MyPySpark:
    def __init__(self, app_name: str = None, session=False, sql=False):
        self.context = SparkContext(master='local', appName=app_name)
        self.session = None
        self.sql = None
        
        if session:
            self.session = SparkSession(self.context)
            
        if sql:
            self.sql = SQLContext(self.context)
            
    def readFile(self, file_path: str, option='csv'):
        if option == 'csv':
            return self.session.read.csv(file_path, inferSchema=True, header=True)
        
        if option == 'json':
            return self.session.read.json(file_path)
        
    def sqlQuery(self, query: str):
        if 'select' in query or 'SELECT' in query:
            return self.sql.sql(query)
        
    def dataframe(self, db_name: str):
        return self.session.table(db_name)