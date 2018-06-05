from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import format_number
from pyspark.ml.feature import StringIndexer

class Data_Adjustments:
    def __init__(self):
       pass

    # public
    def build_df(self, spark, file_location, new_cols=False):
        if new_cols:
            return self.__get_df_with_col_names(
                spark, file_location, new_cols)
        return spark.read.csv(
            file_location, inferSchema=True, header=True)
 
    def drop_columns(self, df, col_names):
        for col in col_names:
            df = df.drop(col)
        return df 

    def vectorize(self, df, col_names):
        assembler = VectorAssembler(
            inputCols=col_names,outputCol="features")
        return assembler.transform(df)

    def string_to_index(self, df, col_names, delete_status=False):
        df = self.__transform_string_to_index(col_names, df)
        if delete_status:
            df = self.drop_columns(df, col_names)
        return df

    def split_data(self, df, weights, seed, col = None):
        if col:
            final_data = df.select('features', col)
            return final_data.randomSplit(weights, seed=seed)
        return df.randomSplit(weights, seed=seed)
    
    #  private
    def __transform_string_to_index(self, col_names, df):
        for col in col_names:
            indexed = StringIndexer(
                inputCol=col, outputCol="{}_category".format(col)).\
                fit(df).transform(df)
            df = indexed
        return df

    def __get_df_with_col_names(
                self, spark, file_location, new_cols):
        data = spark.read.csv(file_location, inferSchema=True)
        return reduce(
            lambda data, idx: data.withColumnRenamed(
            data.schema.names[idx], new_cols[idx]),
            xrange(len(data.schema.names)), data) 

