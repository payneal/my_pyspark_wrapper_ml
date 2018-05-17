import unittest
from binary_classification import Binary_Classification
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from data_adjustments import Data_Adjustments

class Test_binary_classification(unittest.TestCase):
    def setUp(self):
        self.spark = SparkSession.builder.appName(
            "binary_classification").getOrCreate()    
        new_cols = [
            "age", "workclass", "fnlwgt", "education", 
            "education_number","material_status", 
            "occupation", "relationship", "race",
            "sex", "capital_gain", "capital_loss", 
            "hours_per_week","native_country", "50k_status"]
        file_location = "./data/adult.data.csv"
        data = self.spark.read.csv(file_location, inferSchema=True)
        self.df = Data_Adjustments(
            data, file_location, new_cols)
        self.binary_classification = Binary_Classification()
        
    def tearDown(self):
        self.spark.stop()
        self.df = None        
        self.binary_classification = None

    def test_binary_classification(self):
        self.df.drop_columns(['education'])
        self.df.string_to_index([
            "workclass", "material_status", "occupation", "relationship", 'race', 
            "sex", "native_country"], True)
        self.df.show()
        
        self.df.adjust_row_content()


        # create the vector or set things up 
        #self.df.vectorize([
        #    "age", "fnlwgt", "education_number", "capital_gain",
        #    'capital_loss', 'hours_per_week'])
        
        #print "what is this: "
        # output.show()


        # add data to binary classifications
        #self.binary_classification.set_data(self.df.get_df())
         

        #self.binary_classification.split_data(0.6, 0.4, 11)
        #print "what is this: "
        #self.binary_classification.trainning.show()
        self.assertEqual("a", "b")



if __name__ == '__main__':
    unittest.main()
