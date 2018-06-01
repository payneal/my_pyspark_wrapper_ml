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
            "workclass", "material_status", 
            "occupation", "relationship", 'race', 
            "sex", "native_country"], True)
        self.df.vectorize([
            "age", "fnlwgt", "education_number", "capital_gain",
            'capital_loss', 'hours_per_week', 'workclass_category', 
            'material_status_category', 'occupation_category', 
            "relationship_category", "race_category", 
            "sex_category", "native_country_category"])
        train_data, test_data = self.df.split_data([0.6, 0.4], 11)
        # predictions_and_labels = self.binary_classification.get_predictions(
        #    train_data, test_data)

        #print "this is predictions and labels: "
        #print predictions_and_labels
        

        # self.binary_classification.split_data(0.6, 0.4, 11)
        # self.binary_classification.train_algo_to_build_model()
        # self.binary_classification.compute_raw_scores_on_test_data()

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
