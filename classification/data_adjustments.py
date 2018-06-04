from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import format_number
from pyspark.ml.feature import StringIndexer
from pyspark.sql import functions as F

class Data_Adjustments:
    def __init__(self):
       pass

    def drop_columns(self, df, col_names):
        for col in col_names:
            df = df.drop(col)
        return df 

    def vectorize(self, df, col_names):
        assembler = VectorAssembler(
            inputCols=col_names,outputCol="features")
        return assembler.transform(df)

    def string_to_index(self, df, col_names, delete_status=False):
        df = self.transform_string_to_index(col_names, df)
        if delete_status:
            df = self.drop_columns(df, col_names)
        return df

    def transform_string_to_index(self, col_names, df):
        for col in col_names:
            indexed = StringIndexer(
                inputCol=col, outputCol="{}_category".format(col)).\
                fit(df).transform(df)
            df = indexed
        return df

    def split_data(self, df, weights, seed, col = None):
        if col:
            final_data = df.select('features', col)
            return final_data.randomSplit(weights, seed=seed)
        return df.randomSplit(weights, seed=seed)
