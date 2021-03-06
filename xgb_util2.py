import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV


class Model(object):

    def __init__(self, model, features, labels, test_size=0.20, cv_folds=4,
                 early_stopping_rounds=10, num_boost_round=3000,
                 stratify_on_index=False, shuffle=False, random_state=9591029):
        '''Initializes an XGBoost model object with additional methods.
        Creating the object creates a train/test split with the data
        and trains the model. Parameter tuning can be done with
        the  grid_search method. Cross-validation is also available.
        
        Parameters:
        model: XGRegressor model from the XGBoost Scikit-Learn API
        features: pandas DataFrame
            Explanatory features are in columns
        labels: pandas DataFrame 
            Target labels in one column
        test_size: float
            Proportion of input data to hold out
        cv_folds: integer
            Number of cross-validation folds
        early_stopping_rounds: integer
            Tree growing will stop if there is no improvement
             in this number of trees
        num_boost_round: integer
            Maximum number of trees that will be grown 
            
        Returns:
        model object
        '''

        self.test_size = test_size
        self.cv_folds = cv_folds
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.features = features
        self.labels = labels
        self.model = model
        self.stratify_on_index = stratify_on_index
        self.shuffle = shuffle
        self.random_state = random_state
        # assert (stratify_on_index & shuffle), 'Both stratify_on_index and shuffle cannot be true'
        
        self._split()
        self._stop_criteria()
        self.train()

    def _split(self):
        if self.stratify_on_index:
            print('data will be stratified using random state {}'.format(self.random_state))
            self.X_train, self.X_holdout, self.y_train, self.y_holdout = \
            train_test_split(self.features, self.labels,
                             test_size=self.test_size, random_state=self.random_state,
                             stratify=self.features.index)

        elif self.shuffle:
            self.X_train, self.X_holdout, self.y_train, self.y_holdout = \
            train_test_split(self.features, self.labels,
                             test_size=self.test_size, random_state=self.random_state,
                             shuffle=True)

        else:
            self.X_train, self.X_holdout, self.y_train, self.y_holdout = \
            train_test_split(self.features, self.labels,
                             test_size=self.test_size, random_state=self.random_state)

    def _stop_criteria(self):
        self.stop_criteria = {'early_stopping_rounds': self.early_stopping_rounds,
                              'eval_metric': 'rmse',
                              'eval_set': [(self.X_holdout, self.y_holdout)],
                              'verbose': False}

    def _fit(self):
        self.model.fit(self.X_train, self.y_train, **self.stop_criteria)
        self.model.n_estimators = self.model.best_iteration

    def _predict(self):
        self.train_predictions = self.model.predict(self.X_train)
        self.test_predictions = self.model.predict(self.X_holdout)

    def _r2(self):
        self.trainr2 = self.model.score(self.X_train, self.y_train)
        self.testr2 = self.model.score(self.X_holdout, self.y_holdout)
        
    def train(self):
        '''
        Trains the model. You might want to do this if, for example,
        if the parameters have been changed using model.set_params(**{ ... }).
        '''
        self._fit()
        self._predict()
        self._r2()
        
    def do_grid_search(self, params):
        '''
        Performs a grid search based on the dictionary of parameters passed. The
        values in the dictionary are a list of values to try.  The keys are 
        parameters of the model.
        '''
        self.gscv = GridSearchCV(
            self.model, params, n_jobs=-1, cv=self.cv_folds, return_train_score=True)
        self.gscv.fit(self.X_train, self.y_train, **self.stop_criteria)
        self.model = self.gscv.best_estimator_
        self.train    
        
    def _dmatrix(self):
        self.dtrain = xgb.DMatrix(self.features, label=self.labels)
        self.dholdout = xgb.DMatrix(self.X_holdout, label=self.y_holdout)

    def do_model_cross_validation(self):
        '''Runs cross-validation with early stopping.  Changes number of trees
        in the model to best iteration from cross-validation'''
        self._dmatrix()

        xgb_param = self.model.get_xgb_params()
        self.cvresult = xgb.cv(xgb_param, self.dtrain, num_boost_round=self.num_boost_round, nfold=self.cv_folds,
                               metrics='rmse', early_stopping_rounds=self.early_stopping_rounds)
        self.model.n_estimators = self.cvresult.shape[0]
        self.train()
        
    def __str__(self):
        return 'Training R^2 = {:8.3f} \nHold out R^2 = {:8.3f}'.format(self.trainr2, self.testr2)

    def __repr__(self):
        return str(self.model)