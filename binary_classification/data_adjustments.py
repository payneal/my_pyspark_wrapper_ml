from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import format_number
from pyspark.ml.feature import StringIndexer
from pyspark.sql import functions as F

class Data_Adjustments:
    def __init__(self, data, file_location, new_cols):
        self.start_df = reduce(
            lambda data, idx: data.withColumnRenamed(
                data.schema.names[idx], new_cols[idx]),
                xrange(len(data.schema.names)), data) 
        self.df = self.start_df
    
    def get_df(self):
        return self.df

    def show(self):
        umm = self.get_df()
        umm.show()

    def vectorize(self, col_names):
        assembler = VectorAssembler(
            inputCols=col_names,outputCol="features")
        self.df = assembler.transform(self.get_df())

    def drop_columns(self, col_names):
        df = self.get_df()
        for col in col_names:
            df = df.drop(col)
        self.df = df

    def string_to_index(self, col_names, delete_status=False):
        self.df = self.transform_string_to_index(col_names)
        if delete_status:
            self.drop_columns(col_names)

    def transform_string_to_index(self, col_names):
        df  = self.get_df()
        for col in col_names:
            indexed = StringIndexer(
                inputCol=col, outputCol="{}_category".format(col)).\
                fit(df).transform(df)
            df = indexed
        return df

