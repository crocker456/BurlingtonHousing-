#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import quandl
from geopy.distance import great_circle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score # this allows us to split our data
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline # for assembling machine learning pipelines
from sklearn.model_selection import RandomizedSearchCV

def Parse_btv_house():

    # use pandas to read a csv file from Burlington, VT open data
    df = pd.read_csv("https://opendata.arcgis.com/datasets/276ccff527454496ac940b60a2641dda_0.csv")

    # create latitude and longitude columns - drop the old uncessesary column
    locFixes = df['PropertyCenterPoint'].str.strip('()').str.split(', ', expand=True).rename(columns={0:'Latitude',1:'Longitude'})
    df.drop('PropertyCenterPoint', 1, inplace = True)

    # create full and half bath columns - drop the old uncessesary column
    bathFixes = df['Baths'].str.split('/', expand=True).rename(columns={0:'Full_Baths', 1:'Half_Baths'})
    df.drop('Baths', 1, inplace = True)

    # create a dataframe with the new variables - set the correct datatype
    df = pd.concat([df, locFixes,bathFixes],axis=1)
    df['Half_Baths'] = pd.to_numeric(df['Half_Baths'])
    df['Full_Baths'] = pd.to_numeric(df['Full_Baths'])

    #create a year column
    df['SaleDate'] = pd.to_datetime(df['SaleDate'])
    df['Year'] = df['SaleDate'].apply(lambda row: row.year)



    # get Vermont GDP and Housing Price Index from Quandl
    VTGDP = quandl.get("FRED/VTNGSP", start_date="1931-01-01")
    VTHPI = quandl.get("FRED/VTSTHPI", start_date="1931-01-01")

    # Calculate yearly means
    VTGDP = VTGDP.resample('A').mean()
    VTGDP = VTGDP.reset_index()

    # Calculate yearly means
    VTHPI = VTHPI.resample('A').mean()
    VTHPI = VTHPI.reset_index()

    # Create a year columns to join on
    VTGDP['Year'] = VTGDP['Date'].apply(lambda row: row.year)
    VTHPI['Year'] = VTHPI['Date'].apply(lambda row: row.year)

    #rename columns and drop the Date column - It won't be needed
    VTGDP = VTGDP.rename(columns = {'Value': 'VermontGDP'})
    VTHPI = VTHPI.rename(columns = {'Value': 'VermontHPI'})
    VTGDP = VTGDP.drop('Date', 1)
    VTHPI = VTHPI.drop('Date', 1)

    # Join the GDP and HPI data with the BTV housing data
    df = pd.merge(df,VTGDP, 'left', on = 'Year')
    df = pd.merge(df,VTHPI, 'left', on = 'Year')

    df['SaleDate'] = pd.to_datetime(df['SaleDate'])
    modeldf = df.drop(['AccountNumber', 'ParcelID', 'SpanNumber',
           'AlternateNumber', 'Unit', 'CuO1LastName',
           'CuO1FirstName', 'CuO2LastName', 'CuO2FirstName', 'CuO3LastName',
           'CuO3FirstName','LegalReference', 'GrantorLastName', 'FID'], axis = 1)
    modeldf['Latitude'] = pd.to_numeric(modeldf['Latitude'])
    modeldf['Longitude'] = pd.to_numeric(modeldf['Longitude'])
    modeldf['Sale_Year'] = modeldf['SaleDate'].apply(lambda row: row.year)
    modeldf.drop('SaleDate', axis=1, inplace= True)
    modeldf = modeldf[(modeldf['LandUse'] == "Single Family") | (modeldf['LandUse'] == "Residential Condo")]
    modeldf = modeldf[modeldf['SalePrice']> 10000]
    modeldf = modeldf.sort_values('Year')
    modeldf['VermontGDP'] = modeldf['VermontGDP'].fillna(method='ffill')
    modeldf['VermontHPI'] = modeldf['VermontHPI'].fillna(method='ffill')
    modeldf = modeldf.dropna()



    # convert latitude and longitude to numeric
    modeldf['Latitude'] = pd.to_numeric(modeldf['Latitude'])
    modeldf['Longitude'] = pd.to_numeric(modeldf['Longitude'])

    #loop over these columns and create a list of the differences between each observation and city hall
    distances = []
    for i, j in zip(modeldf['Latitude'], modeldf['Longitude']):
        val = (i, j)
        Cityhall = (44.47647568031712, -73.21353835752235)
        dist = great_circle(val,Cityhall).miles
        distances.append(dist)
    modeldf['Distances'] = distances
    return modeldf

#basictransformer modified from https://colab.research.google.com/drive/1yHnTLJVWDzI7_WqjTlgRMBOTcwt8qIr1#scrollTo=VD_8-0VObs7B&forceEdit=true&offline=true&sandboxMode=true


class BasicTransformer(BaseEstimator):

    def __init__(self, cat_threshold=None, num_strategy='median', return_df=False):
        # store parameters as public attributes
        self.cat_threshold = cat_threshold

        if num_strategy not in ['mean', 'median']:
            raise ValueError('num_strategy must be either "mean" or "median"')
        self.num_strategy = num_strategy
        self.return_df = return_df



    def fit(self, X, y=None):
        # Assumes X is a DataFrame
        self._columns = X.columns.values

        # Split data into categorical and numeric
        self._dtypes = X.dtypes.values
        self._kinds = np.array([dt.kind for dt in X.dtypes])
        self._column_dtypes = {}
        is_cat = self._kinds == 'O'
        self._column_dtypes['cat'] = self._columns[is_cat]
        self._column_dtypes['num'] = self._columns[~is_cat]
        self._feature_names = self._column_dtypes['num']
        # it is essential to use one of Sklearn's scalers, otherwise you cannnot predict for single points
        # this is the only change that I have made from   https://colab.research.google.com/drive/1yHnTLJVWDzI7_WqjTlgRMBOTcwt8qIr1#scrollTo=VD_8-0VObs7B&forceEdit=true&offline=true
        self._scaler = StandardScaler()
        # Create a dictionary mapping categorical column to unique values above threshold
        self._cat_cols = {}
        for col in self._column_dtypes['cat']:
            vc = X[col].value_counts()
            if self.cat_threshold is not None:
                vc = vc[vc > self.cat_threshold]
            vals = vc.index.values
            self._cat_cols[col] = vals
            self._feature_names = np.append(self._feature_names, col + '_' + vals)

        # get total number of new categorical columns
        self._total_cat_cols = sum([len(v) for col, v in self._cat_cols.items()])

        # get mean or median
        self._num_fill = X[self._column_dtypes['num']].agg(self.num_strategy)
        self._scaler.fit(X[self._column_dtypes['num']])
        return self


    def transform(self, X):
        # check that we have a DataFrame with same column names as the one we fit
        if set(self._columns) != set(X.columns):
            raise ValueError('Passed DataFrame has different columns than fit DataFrame')
        elif len(self._columns) != len(X.columns):
            raise ValueError('Passed DataFrame has different number of columns than fit DataFrame')

        # fill missing values
        X_num = X[self._column_dtypes['num']].fillna(self._num_fill)



        X_num = self._scaler.transform(X_num)

        # create separate array for new encoded categoricals
        X_cat = np.empty((len(X), self._total_cat_cols), dtype='int')
        i = 0
        for col in self._column_dtypes['cat']:
            vals = self._cat_cols[col]
            for val in vals:
                X_cat[:, i] = X[col] == val
                i += 1


        # concatenate transformed numeric and categorical arrays
        data = np.column_stack((X_num, X_cat))

        # return either a DataFrame or an array
        if self.return_df:
            return pd.DataFrame(data=data, columns=self._feature_names)
        else:
            return data

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def get_feature_names(self):
        return self._feature_names

def fit_validate_bayes_ridge():

    df = Parse_btv_house()
    # outcome variable is SalePrice
    y = df['SalePrice']
    # predict outcome variable with all other variables
    X = df.drop('SalePrice', 1)
    # use sklearn train_test_split to obtain training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X.copy(), y.copy())


    tune_blm = BayesianRidge()
    bt = BasicTransformer(cat_threshold=3, return_df=True)

    basic_pipe = Pipeline([('bt', bt), ('bayes_ridge', tune_blm)])
    param_dist = {'bayes_ridge__fit_intercept': [True, False],
                   'bayes_ridge__alpha_1': [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-1, 1, 5, 10],
                   'bayes_ridge__alpha_2': [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-1, 1, 5, 10],
                   'bayes_ridge__lambda_1': [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-1, 1, 5, 10],
                   'bayes_ridge__lambda_2': [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-1, 1, 5, 10]
                   }


    # run randomized search
    n_iter = 20
    random_search = RandomizedSearchCV(basic_pipe, cv = 10, param_distributions=param_dist,
                                       n_iter=n_iter, n_jobs=-1, scoring = 'neg_mean_squared_error')
    random_search.fit(X_train, y_train)
    bestmodel = random_search.best_estimator_
    best_ridge_params = {}
    for i,j in random_search.best_params_.items():
        best_ridge_params[i.split('_')[3]+'_'+i.split('_')[4]] = j
    final_ridge = BayesianRidge(**best_ridge_params)
    basic_pipe = Pipeline([('bt', bt), ('bayes_ridge', final_ridge)])
    basic_pipe.fit(X_train, y_train)
    scores = cross_val_score(basic_pipe, X_train, y_train, cv = 10)
    score = basic_pipe.score(X_test, y_test)
    import datetime
    d = datetime.datetime.today()
    d = str(d).split(" ")[0]
    import os
    # define the name of the directory to be created
    path = 'ValidationPlots_{}'.format(d)
    try:
        os.mkdir(path)
        print('Directory ValidationPlots_{} created'.format(d))
    except FileExistsError:
        print('Directory already exists!')

    pred, sd = basic_pipe.predict(X_test, return_std = True)

    for i in range(50):
        numbers = range(len(y_test))
        number= np.random.choice(numbers, 1)
        num = []
        for i, j in enumerate(y_test):
            if i == number:
                num.append(j)
            else:
                pass
        X = np.random.normal(pred[number], sd[number], 100000)
        price = 450000
        prob = len(X[X > price])/ len(X) * 100
        plt.figure(figsize = (12, 6))
        sns.distplot(X)
        plt.axvline(num, label = 'True Value = {}'.format(int(num[0])), color = 'C1')
        plt.axvline(pred[number], label = 'Posterior Predictive Mean = {}'.format(int(pred[number])))
        plt.legend()
        plt.title('ID = {} \n Residual: {} \n Model R^2: {}'.format(int(number), int(num - pred[number]),round(score,4)))
        plt.savefig("ValidationPlots_{}/X_testID={}".format(d,int(number)))
        plt.close()

#if __name__ == "__main__":
#    fit_validate_bayes_ridge()
