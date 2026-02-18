from sklearn.ensemble import RandomForestRegressor

#X-> Dazzler parameters
#Y -> Pulse shape
class ROUNDmodelRF:
    def __init__(self,X_train,X_test,y_train,y_test,
                n_estimators = 300, max_features = 1.0, max_depth=20, random_state=18):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test

        self.n_estimators=n_estimators
        self.max_features=max_features
        self.max_depth=max_depth
        self.random_state=random_state

    def train(self):
        #train forward mdoel
        print('Training the Random Forest forward model',)
        self.rf_model_fwd=RandomForestRegressor(n_estimators=self.n_estimators,
                                            max_features=self.max_features,
                                            max_depth=self.max_depth,
                                            random_state=self.random_state)
        self.rf_model_fwd.fit(self.X_train,self.y_train)

        #train backward model
        print('Training the Random Forest backward model',)
        self.rf_model_bwd=RandomForestRegressor(n_estimators=self.n_estimators,
                                            max_features=self.max_features,
                                            max_depth=self.max_depth,
                                            random_state=self.random_state)
        self.rf_model_bwd.fit(self.y_train,self.X_train)
    def predict(self,y_data):
        #backward prediction
        self.X_predict=self.rf_model_bwd.predict(y_data)
        #forward prediction
        self.y_predict=self.rf_model_fwd.predict(self.X_predict)
        return self.y_predict
    def error_calc(self):
        print('Calculating the Mean Absolute Error')
        #Reset the index on y_test to have the same indexes as y_predict
        y_test_reset=self.y_test.reset_index(drop=True)
        #study the erro distribution
        mae_error=abs(y_test_reset-self.y_predict)
        mae_error=mae_error.sum(axis=1)/self.y_test.shape[1] #sum error / num columns
        return mae_error
    
