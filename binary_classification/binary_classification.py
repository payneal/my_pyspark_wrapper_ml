from pyspark.mllib.classification import LogisticRegressionWithLBFGS

class Binary_Classification:
    def __init__(self):
        self.df = None
        self.training = None
        self.test = None
        
    def set_data(self, dataframe):
        print "this is type: {}".format(type(dataframe))
        # dataframe.show() 

    def split_data(self, part_one, part_two, the_seed=None):
        if the_seed: 
            self.training, self.test =self.df.randomSplit([
                part_one, part_two], seed=the_seed) 
        else:
            self.training, self.test = self.df.randomSplit([
                part_one, part_two])
        
