# ml test casses 

from pyspark import SparkConf
from pyspark.sql import SparkSession
import unittest

from pyspark_wrapper import Pyspark_Wrapper

class Test_Easy_ML(unittest.TestCase):

    def setUp(self):
        # create a single node Spark application
        conf = SparkConf()
        conf.set("spark.executor.memory", "1g")
        conf.set("spark.cores.max", "1")
        conf.set("spark.app.name", "nosetest")
        SparkSession._instantiatedContext = None
        self.spark = SparkSession.builder.config(
            conf=conf).getOrCreate()

        self.sc = self.spark.sparkContext
        self.mock_df = self.mock_data()

    def tearDown(self):
        self.sc.stop()
        self.spark.stop()

    def mock_data(self):
        """Mock data to imitate read from database."""
        mock_data_rdd = self.sc.parallelize([(
            "A", 1, 1), ("B", 1, 0),
            ("C", 0, 2), ("D", 2, 4),("E", 3, 5)])

        schema = ["id", "x", "y"]
    
        mock_data_df = self.spark.createDataFrame(
            mock_data_rdd, schema)
    
        return mock_data_df

    def test_count(self):
        self.assertEqual(len(self.mock_df.collect ()), 5)


    def test_hello(self): 
        x = Pyspark_Wrapper()
        self.assertEqual(x.hello(), "hello world")

if __name__ == '__main__':
    unittest.main()
