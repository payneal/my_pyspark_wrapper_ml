import unittest
from data_adjustments import Data_Adjustments
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

class Test_Data_Adjustments(unittest.TestCase):
    def setUp(self):
        self.spark = SparkSession.builder.appName(
            "data_adjustments").getOrCreate()
        new_cols = [
            "age", "workclass", "fnlwgt", "education", 
            "education_number","material_status", 
            "occupation", "relationship", "race",
            "sex", "capital_gain", "capital_loss", 
            "hours_per_week","native_country", "50k_status"]
        file_loc = "./data/adult.data.csv"
        data = self.spark.read.csv(file_loc, inferSchema=True) 
        self.df = Data_Adjustments(data, file_loc, new_cols)

    def tearDown(self):
        self.spark.stop()
        self.df = None
        self.data = None
        self.data_adjustment = None

    def test_test_droping_a_column_on_data_frame(self):
        df = self.df.get_df()
        all_col_names = df.schema.names
        self.df.drop_columns(['education'])
        df = self.df.get_df()
        current_col_names = df.schema.names
    
        self.assertNotEqual(all_col_names, current_col_names)
        self.assertNotIn('education', current_col_names)

    def test_string_to_index_on_row_data_no_delete(self):
        df = self.df.get_df()
        row_data_1 = df.first()['workclass']
        self.df.string_to_index(['workclass'])
        df = self.df.get_df()
        row_data_2 = df.first()['workclass_category']
        
        self.assertIn('workclass_category', df.schema.names)
        self.assertNotEqual(row_data_1, row_data_2)
        self.assertNotEqual(type(row_data_1), type(row_data_2))

    def test_string_to_index_on_row_data_with_delete(self):
        df = self.df.get_df()
        self.df.string_to_index(['workclass'], True)
        df = self.df.get_df()
        self.assertNotIn('workclass', df.schema.names)
        self.assertIn('workclass_category', df.schema.names)

    def test_vectorizing_data_frame(self):
        self.df.vectorize(["age", "fnlwgt"])
        df = self.df.get_df()
        self.assertIn('features', df.schema.names)
        self.assertEqual(len(df.first()['features']), 2)

    def test_spliting_data(self):
        self.df.drop_columns(['education'])
        self.df.string_to_index([
            "workclass", "material_status", "occupation",
            "relationship", "race", "sex", "native_country"], True)
        self.df.vectorize([
            "age", "fnlwgt", "education_number", "capital_gain",
            "capital_loss", "hours_per_week", "workclass_category",
            "material_status_category", "occupation_category",
            "relationship_category", "race_category", "sex_category",
            "native_country_category"])
    
        df_count = self.df.get_df().count()
        train_data, test_data = self.df.split_data([0.6, 0.4], 11)
        train_count = train_data.count()
        test_count = test_data.count()
        self.assertEqual( train_count + test_count, df_count)

        # print "this is train data count: {}".format(train_data.count())
        # print "this is test data count: {}".format(test_data.count())
        # print "look at this: "        
        # train_data.show()
        # print "look at this: "
        # test_data.show()
    

if __name__ == '__main__':
    unittest.main()

