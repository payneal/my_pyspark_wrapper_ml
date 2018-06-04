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
        self.df = reduce(
            lambda data, idx: data.withColumnRenamed(
            data.schema.names[idx], new_cols[idx]),
            xrange(len(data.schema.names)), data) 
        self.data_adjustments = Data_Adjustments()

    def tearDown(self):
        self.spark.stop()
        self.df = None
        self.data = None
        self.data_adjustments = None

    def test_test_droping_a_column_on_data_frame(self):
        all_col_names = self.df.schema.names
        df = self.data_adjustments.drop_columns(self.df, ['education'])
        current_col_names = df.schema.names
        self.assertNotEqual(all_col_names, current_col_names)
        self.assertNotIn('education', current_col_names)

    def test_string_to_index_on_row_data_no_delete(self):
        row_data_1 = self.df.first()['workclass']
        df = self.data_adjustments.string_to_index(self.df, ['workclass'])
        row_data_2 = df.first()['workclass_category']
        self.assertIn('workclass_category', df.schema.names)
        self.assertNotEqual(row_data_1, row_data_2)
        self.assertNotEqual(type(row_data_1), type(row_data_2))

    def test_string_to_index_on_row_data_with_delete(self):
        df = self.data_adjustments.string_to_index( self.df,['workclass'], True)
        self.assertNotIn('workclass', df.schema.names)
        self.assertIn('workclass_category', df.schema.names)

    def test_vectorizing_data_frame(self):
        df = self.data_adjustments.vectorize(self.df, ["age", "fnlwgt"])
        self.assertIn('features', df.schema.names)
        self.assertEqual(len(df.first()['features']), 2)

    def test_spliting_data(self):
        df = self.data_adjustments.drop_columns(self.df, ['education'])
        df = self.data_adjustments.string_to_index(df, [
            "workclass", "material_status", "occupation",
            "relationship", "race", "sex", "native_country"], True)
        df = self.data_adjustments.vectorize(
            df, ["age", "fnlwgt", "education_number", "capital_gain",
            "capital_loss", "hours_per_week", "workclass_category",
            "material_status_category", "occupation_category",
            "relationship_category", "race_category", "sex_category",
            "native_country_category"])
        df_count = df.count()
        train_data, test_data = self.data_adjustments.split_data(
            df,[0.6, 0.4], 11)
        train_count = train_data.count()
        test_count = test_data.count()
        self.assertEqual( train_count + test_count, df_count)
    
if __name__ == '__main__':
    unittest.main()
