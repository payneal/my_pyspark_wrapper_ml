import unittest
from classification import Classification
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from data_adjustments import Data_Adjustments

class Test_classification(unittest.TestCase):
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
        self.df = reduce(
            lambda data, idx: data.withColumnRenamed(
            data.schema.names[idx], new_cols[idx]),
            xrange(len(data.schema.names)), data) 
        self.data_adjustments = Data_Adjustments()
        self.classification = Classification()
        
    def tearDown(self):
        self.spark.stop()
        self.df = None
        self.data_adjustment = None        
        self.classification = None

    def test_binary_classification(self):
        print "this is start df"
        print self.df.columns
        df = self.data_adjustments.drop_columns(self.df, ['education'])
        df = self.data_adjustments.string_to_index(df, [
            "workclass", "material_status", 
            "occupation", "relationship", 'race', 
            "sex", "native_country"], True)
        df = self.data_adjustments.vectorize(
            df, [ "age", "fnlwgt", "education_number", "capital_gain",
            'capital_loss', 'hours_per_week', 'workclass_category', 
            'material_status_category', 'occupation_category', 
            "relationship_category", "race_category", 
            "sex_category", "native_country_category"])
        train_data, test_data = self.data_adjustments.split_data(
            df, [0.6, 0.4], 11, "capital_gain")
        predictions_and_labels = self.classification.get_predictions(
            train_data, test_data, "capital_gain")

        print "this is predictions and labels: "
        print predictions_and_labels
        

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
        print df.columns
        self.assertEqual("a", "b")



if __name__ == '__main__':
    unittest.main()
