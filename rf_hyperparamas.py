from sklearn import model_selection
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, model_selection, metrics, ensemble

from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

import dat

from math import floor

class MlpHyperparametersTest:

    NUM_FOLDS = 5

    def __init__(self, randomSeed):

        self.randomSeed = randomSeed
        self.initDataset()
        #self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS, random_state=self.randomSeed)
        self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS, random_state=self.randomSeed, shuffle=True)

    def initDataset(self):
        x_train, x_test, y_train, y_test=dat.x_train, dat.x_test, dat.y_train, dat.y_test

        self.X = x_train
        self.y = y_train
        
        



    #   n_estimators :int
    #   max_depth:int
    #    criterion:  {'gini','entropy', 'log_loss'}
    #   max_features  :{'sqrt','log2','None'}
    #        bootstrap:{'false' ,'true'}

    
    def convertParams(self, params):
      
        # transform the layer sizes from float (possibly negative) values into hiddenLayerSizes tuple:
       
        n_estimators = floor(params[0])
        max_depth = floor(params[1])
        criterion = ['gini','entropy', 'log_loss'][floor(params[2])]
        max_features = ['sqrt','log2',None][floor(params[3])]
        bootstrap = [True,False][floor(params[4])]
        print( n_estimators, max_depth ,criterion ,max_features ,bootstrap)

        return n_estimators, max_depth ,criterion ,max_features ,bootstrap
    

    @ignore_warnings(category=ConvergenceWarning)
    def getAccuracy(self, params):
        n_estimators, max_depth ,criterion ,max_features ,bootstrap = self.convertParams(params)

        self.classifier = ensemble.RandomForestClassifier(random_state=self.randomSeed,
                                        n_estimators=n_estimators,
                                        max_depth=max_depth ,
                                        criterion=criterion ,
                                        max_features=max_features ,
                                        bootstrap=bootstrap)
        
                                        

        cv_results = model_selection.cross_val_score(self.classifier,
                                                     self.X,
                                                     self.y,
                                                     cv=self.kfold,
                                                     scoring='accuracy')

        return cv_results.mean()

    def formatParams(self, params):
        n_estimators, max_depth ,criterion ,max_features ,bootstrap = self.convertParams(params)
        return "'n_estimators'={}\n " \
               "'max_depth'='{}'\n " \
               "'criterion'='{}'\n " \
               "'max_features'={}\n " \
               "'bootstrap'='{}'"\
            .format(n_estimators, max_depth ,criterion ,max_features ,bootstrap)
