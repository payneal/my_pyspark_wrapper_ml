from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import OneVsRest
from pyspark.ml.classification import NaiveBayes

class Classification:
    def __init__(self):
        pass

    # use logistic  regression when predicting a yes/no, true/false, etc 2 options
    def get_logistic_regression(self, train_data, test_data, col_to_check):
        
        print "----------------------------------------------------------------"
        print ""
        info = {}
        lr_churn  = LogisticRegression(labelCol=col_to_check)
        
        print  "this is  lr_churn: {}".format(lr_churn) 
        print "this is lr_churn type: {}".format(type(lr_churn)) 

        fitted_churn_model = lr_churn.fit(train_data)
        print "this is fitted_churn_model: {}".format(fitted_churn_model)
        print "this is fitted_churn_model type: {}".format(
            type(fitted_churn_model))

        print "what is here:{}".format(dir( fitted_churn_model) )

        trainning_sum = fitted_churn_model.summary
        print "trainning sum: {}".format(trainning_sum)
        print "trainning sum type: {}".format(type(trainning_sum))

        trainning_sum.predictions.describe().show()
        
        print "should show the predictions above"

        predictions_and_labels  = fitted_churn_model.evaluate(test_data)

        print "this is fitted_churn_model: {}".format(predictions_and_labels)
        print "this si fitted_chur_model after evaluate on train data type: {}".format(
            type(predictions_and_labels))


        print ""
        print "showing predictions  and labels count: "
        print predictions_and_labels.predictions.show(351)
        print "----------------------------------------------------------------"
        print ""
        return predictions_and_labels

