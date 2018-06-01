from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.classification import LogisticRegression


class Binary_Classification:
    def __init__(self):
        self.df = None
        self.training = None
        self.test = None
        self.model = None

    def set_data(self, dataframe):
        self.df = dataframe

    def get_predictions(self, train_data, test_data):
        
        lr_churn  = LogisticRegression(labelCol='features')

        fitted_churn_model = lr_churn.fit(train_data)
        
        trainning_sum = fitted_churn_model.summary

        trainning_sum.predictions.describe().show()
        
        return fitted_churn_model.evaluate(test_churn)


    # def compute_raw_scores_on_test_data(self):
    #    predictionAndLabels = self.test.map(
    #        lambda lp: (float(self.model.predict(lp.features)), lp.label))
    #    metrics = BinaryClassificationMetrics(predictionAndLabels)

    #    print "what is prediction: {}".format(predictionAndLabels)
    #    print "what is metrics: {}".format(metrics)
