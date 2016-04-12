import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.grid.grid_search import H2OGridSearch
import pandas as pd
#from sklearn import grid_search
import numpy
from sklearn.metrics import roc_auc_score

h2o.init()
df = h2o.import_file(path='ds1.100.csv')
df.columns[100] = 'y'
train, test = df.split_frame(ratios=[0.75], seed=0)
logistic = H2OGeneralizedLinearEstimator(solver='IRLSM', family='binomial',
	       alpha=1, lambda_search=True)
logistic.train(x=range(0,100), y=100, training_frame=train, 
	validation_frame=test)
#AUC: 0.924031049047

logistic_off = H2OGeneralizedLinearEstimator(solver='IRLSM', family='binomial',
	       alpha=1, lambda_search=True)
logistic_off.train(x=range(0,100), y=100, training_frame=train, 
	validation_frame=test, offset_column='C1')


'''
clf = grid_search.GridSearchCV(logistic, parameters)
clf.fit(train[0:100], train[101], scoring=auc)
'''

parameters = {'lambda':list(numpy.arange(0.000001, 0.0002, 0.000005))}
#parameters = {'lambda':[0.00001, 0.00002]}
gs = H2OGridSearch(H2OGeneralizedLinearEstimator(family='binomial', alpha=1), parameters)
gs.train(x=range(0,100), y=100, training_frame=train)
gs.auc()

logistic_gs = H2OGeneralizedLinearEstimator(solver='IRLSM', family='binomial',
	       alpha=1, Lambda=0.000026)
logistic_gs.train(x=range(0,100), y=100, training_frame=train, validation_frame=test)
#AUC: 0.924354857121


gbt = H2OGradientBoostingEstimator(ntrees = 500, max_depth=6, 
	learn_rate=0.1)
gbt.train(x=range(0,100), y=100, training_frame=train, 
	validation_frame=test)
pred = gbt.predict(x=range(0,100), test)
roc_auc_score(h2o.as_list(test[100], use_pandas=True)['C101'], h2o.as_list(pred, use_pandas=True)['predict'])
# auc: 0.89579468056765887

parameters = {'ntrees':range(300, 2000, 300),
              'max_depth':[4, 6, 8],
              'learn_rate':[0.001, 0.01, 0.1, 0.2]}
criteria = {'strategy': 'RandomDiscrete', 'max_models': 100, 'max_runtime_secs': 28800}
gs_gbt = H2OGridSearch(H2OGradientBoostingEstimator(), parameters, criteria)
gs_gbt.train(x=range(0,100), y=100, training_frame=train, validation_frame=test)





