from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.classification import LogisticRegression


class Classification:
    def __init__(self):
        pass


    def get_predictions(self, train_data, test_data, col_to_check):

        lr_churn  = LogisticRegression(labelCol=col_to_check)

        print "afterlogistic regression"

        print  "this is  lr_churn: {}".format(lr_churn) 

        fitted_churn_model = lr_churn.fit(train_data)

        print "after fitted churn model"

        print "this is fitted_churn_model: {}".format(fitted_churn_model)

        trainning_sum = fitted_churn_model.summary

        print "after trainning sum"

        trainning_sum.predictions.describe().show()
        
        print "should show the predictions above"

        return fitted_churn_model.evaluate(test_churn)


    # def compute_raw_scores_on_test_data(self):
    #    predictionAndLabels = self.test.map(
    #        lambda lp: (float(self.model.predict(lp.features)), lp.label))
    #    metrics = BinaryClassificationMetrics(predictionAndLabels)

    #    print "what is prediction: {}".format(predictionAndLabels)
    #    print "what is metrics: {}".format(metrics)
