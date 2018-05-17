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
        df  = self.get_df()
        for col in col_names:
            indexed = StringIndexer(inputCol=col, outputCol="{}_category".format(col))
            indexed = indexed.fit(df).transform(df)
            df = indexed
        self.df = df

        if delete_status:
            self.drop_columns(col_names)

    def adjust_row_content(self):
        df = self.get_df()

        umm = df.head(1)
        
        x = umm[0]['50k_status']

        if x == "<=50k":
            print "its less than or == to 50k"
        else:
            print "its greater thann 50k"

        # df.select(df['50k_status'], F.when(df['50k_status'] == '<=50k', 1).otherwise(0)).show()
    

    # def cast_column_for_vector(self, data,  col_name, cast_to_col_type, digit_length=None):
    #     if digit_length:
    #         format_number(data[col_name].cast(
    #             cast_to_col_type), digit_length).alias(col_name)
    #    else:
    #        data[col_name].cast(cast_to_col_type).alias(col_name)
    #    return data
