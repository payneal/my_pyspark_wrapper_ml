import unittest
from classification import Classification
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from data_adjustments import Data_Adjustments

class Test_classification(unittest.TestCase):
    def setUp(self):
        self.spark = SparkSession.builder.appName(
            "classification").getOrCreate()    
        self.data_adjustments = Data_Adjustments()
        self.classification = Classification()
        
    def tearDown(self):
        self.spark.stop()
        self.data_adjustment = None        
        self.classification = None

    def build_customer_churn_df(self):
        file_loc = "./data/customer_churn.csv"
        return self.data_adjustments. build_df(
            self.spark, file_loc)

    def test_classification_using_logistic_regression(self):
        df  = self.build_customer_churn_df()
        df.show(351) 
        df = self.data_adjustments.vectorize(
            df, ["Age", "Total_Purchase", "Account_Manager", 
            "Years", "Num_Sites"])
        train_data, test_data = self.data_adjustments.split_data(
            df, [0.6, 0.4], 11, "churn")
        
        logistic_regression = self.classification.get_logistic_regression(
            train_data, test_data, "churn")

        print "this is predictions and labels:i "
        print logistic_regression
        self.assertEqual("a", "b")

    # def test_classification_using_decision_tree(self):
    #   self.assertEqual("a", "b")
      
    # def test_classification_using_random_forest(self):
    #   self.assertEqual("a", "b")

    # def test_classification_using_gbt(self):
    #   self.assertEqual("a", "b")

    # def test_classification_using_multilayer_perceptron(self):
    #   self.assertEqual("a", "b")

    # def test_classification_using_linearSVC(self):
    #   self.assertEqual("a", "b")

    # def test_classification_using_one_vs_Rest(self):
    #   self.assertEqual("a", "b")

    # def test_classification_using_naive_bayes(self):
    #   self.assertEqual("a", "b")

if __name__ == '__main__':
    unittest.main()
