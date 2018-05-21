from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.evaluation import BinaryClassificationMetrics

class Binary_Classification:
    def __init__(self):
        self.df = None
        self.training = None
        self.test = None
        self.model = None

    def set_data(self, dataframe):
        self.df = dataframe

    def split_data(self, part_one, part_two, the_seed=None):
        if the_seed: 
            self.training, self.test =self.df.randomSplit([
                part_one, part_two], seed=the_seed) 
        else:
            self.training, self.test = self.df.randomSplit([
                part_one, part_two])
        self.training.cache()
    
    # def train_algo_to_build_model(self):
    #     self.model = LogisticRegressionWithLBFGS.train(self.training)

    # def compute_raw_scores_on_test_data(self):
    #    predictionAndLabels = self.test.map(
    #        lambda lp: (float(self.model.predict(lp.features)), lp.label))
    #    metrics = BinaryClassificationMetrics(predictionAndLabels)

    #    print "what is prediction: {}".format(predictionAndLabels)
    #    print "what is metrics: {}".format(metrics)
