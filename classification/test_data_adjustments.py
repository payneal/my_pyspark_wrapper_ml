import unittest
from data_adjustments import Data_Adjustments
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

class Test_Data_Adjustments(unittest.TestCase):
    def setUp(self):
        self.spark = SparkSession.builder.appName(
            "data_adjustments").getOrCreate()
        self.data_adjustments = Data_Adjustments()

    def tearDown(self):
        self.spark.stop()
        self.data_adjustments = None


    def build_adult_df(self):
        new_cols = [
            "age", "workclass", "fnlwgt", "education", 
            "education_number","material_status", 
            "occupation", "relationship", "race",
             "sex", "capital_gain", "capital_loss", 
            "hours_per_week","native_country", "50k_status"]
        file_loc = "./data/adult.data.csv"
        return self.data_adjustments.build_df(
            self.spark, file_loc, new_cols)

    def test_building_data_fram_adding_cols_names(self):
        df = self.build_adult_df()
        self.assertEqual(
            str(type(df)), 
            "<class 'pyspark.sql.dataframe.DataFrame'>")

    def test_building_data_frame_infer_and_header_true(self):
        file_loc = "./data/customer_churn.csv"
        df = self.data_adjustments.build_df(
            self.spark, file_loc)
        self.assertEqual(
            str(type(df)), 
            "<class 'pyspark.sql.dataframe.DataFrame'>")

    def test_droping_a_column_on_data_frame(self):
        df = self.build_adult_df()
        all_col_names = df.schema.names
        df = self.data_adjustments.drop_columns(
            df, ['education'])
        current_col_names = df.schema.names
        self.assertNotEqual(all_col_names, current_col_names)
        self.assertNotIn('education', current_col_names)

    def test_string_to_index_on_row_data_no_delete(self):
        df = self.build_adult_df()
        row_data_1 = df.first()['workclass']
        df = self.data_adjustments.string_to_index(
            df, ['workclass'])
        row_data_2 = df.first()['workclass_category']
        self.assertIn('workclass_category', df.schema.names)
        self.assertNotEqual(row_data_1, row_data_2)
        self.assertNotEqual(type(row_data_1), type(row_data_2))

    def test_string_to_index_on_row_data_with_delete(self):
        
        df = self.build_adult_df()
        df = self.data_adjustments.string_to_index( 
            df,['workclass'], True)
        self.assertNotIn('workclass', df.schema.names)
        self.assertIn('workclass_category', df.schema.names)

    def test_vectorizing_data_frame(self):
        df = self.build_adult_df()

        df = self.data_adjustments.vectorize(
            df, ["age", "fnlwgt"])
        self.assertIn('features', df.schema.names)
        self.assertEqual(len(df.first()['features']), 2)

    def test_spliting_data(self):
        
        df = self.build_adult_df()
        df = self.data_adjustments.drop_columns(
            df, ['education'])
        df = self.data_adjustments.string_to_index(
            df, [ "workclass", "material_status", "occupation",
            "relationship", "race", "sex", "native_country"],
            True)
        df = self.data_adjustments.vectorize(
            df, ["age", "fnlwgt", "education_number", 
            "capital_gain", "capital_loss", "hours_per_week", 
            "workclass_category", "material_status_category", 
            "occupation_category","relationship_category", 
            "race_category", "sex_category",
            "native_country_category"])
        df_count = df.count()
        train_data, test_data = self.data_adjustments.split_data(
            df,[0.6, 0.4], 11)
        train_count = train_data.count()
        test_count = test_data.count()
        self.assertEqual( train_count + test_count, df_count)
    
if __name__ == '__main__':
    unittest.main()
