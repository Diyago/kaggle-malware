import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import sklearn.metrics as metrics
from models_zoo import BesXGboost, BesLightGBM, BesCatBoost
from target_encoding import TargetEncoding
import gc
import time
import matplotlib as plt
import joblib
import seaborn as sns
import lightgbm as lgb
from numba import jit
from pathlib import Path

# fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
@jit
def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc

def eval_auc(preds, dtrain):
    labels = dtrain.get_label()
    return 'auc', fast_auc(labels, preds), True

class Kaggle:
    """
    Class skeleton for classification

        * Data Preparation
        * Feature Engineering
        * Models Stacking
    """

    def __init__(self, data_path, metric='auc', mode=0):
        self.data_path = data_path
        self.mode = mode
        self.df_train = None
        self.df_test = None
        self.target_colname = None
        self.id_colname = None
        self.folds  = None
        self.sep = None
        self.metric = metric
        self.compute_metric, self.maximize = self.get_metric(metric)
        # TODO store big list of params on separate file
        self.dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'float16',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float16',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int8',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'float16',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'category',
        'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float16',
        'SmartScreen':                                          'category',
        'Firewall':                                             'float16',
        'UacLuaenable':                                         'float32',
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float16',
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float16',
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float32',
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float32',
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float32',
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'int16',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int16',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        'Census_IsFlightingInternal':                           'float16',
        'Census_IsFlightsDisabled':                             'float16',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float16',
        'Census_FirmwareVersionIdentifier':                     'float32',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'float16',
        'Census_IsVirtualDevice':                               'float16',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
        'Wdft_IsGamer':                                         'float16',
        'Wdft_RegionIdentifier':                                'float16',
        'HasDetections':                                        'int8'
        }

    @staticmethod
    def get_metric(metric):
        """Returns metric evaluation"""
        if metric == 'auc':
            return metrics.roc_auc_score, True
        if metric == 'mae':
            return metrics.mean_absolute_error, False

    def read_train_data(self, train_name='train.csv', sep=',', target_colname=None, id_colname=None, num_rows = None):
        self.sep = sep
        self.df_train = pd.read_csv(os.path.join(self.data_path, train_name), sep=self.sep, dtype=self.dtypes, nrows = num_rows)
        if 5244810 in self.df_train.index:
            self.df_train.loc[5244810,'AvSigVersion'] = '1.273.1144.0'
            self.df_train['AvSigVersion'].cat.remove_categories('1.2&#x17;3.1144.0',inplace=True)
        self.df_train['AvSigVersion_1'] = self.df_train['AvSigVersion'].map(lambda x: np.int(x.split('.')[1]))
       # datedictAS = np.load(os.path.join(self.data_path, 'AvSigVersionTimestamps.npy'))[()]

        #df_train['DateAS'] = df_train['AvSigVersion'].map(datedictAS)

        self.df_train = self.df_train[self.df_train['AvSigVersion_1']>=250 ]
        
        if target_colname:
            self.target_colname = target_colname
        if id_colname:
            self.id_colname = id_colname

    def read_test_data(self, test_name='test.csv'):
        self.df_test = pd.read_csv(os.path.join(self.data_path, test_name), sep=self.sep, dtype=self.dtypes)

    @staticmethod
    def reduce_mem_usage(data, verbose = True):
        numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = data.memory_usage().sum() / 1024**2
        if verbose:
            print('Memory usage of dataframe: {:.2f} MB'.format(start_mem))
        
        for col in data.columns:
            col_type = data[col].dtype
            
            if col_type in numerics:
                c_min = data[col].min()
                c_max = data[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        data[col] = data[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        data[col] = data[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        data[col] = data[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        data[col] = data[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        data[col] = data[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        data[col] = data[col].astype(np.float32)
                    else:
                        data[col] = data[col].astype(np.float64)
        
        end_mem = data.memory_usage().sum() / 1024**2
        if verbose:
            print('Memory usage after optimization: {:.2f} MB'.format(end_mem))
            print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        
        return data

    def create_validation_split(self, n_folds=5, stratified=False):
        self.folds = n_folds
        if Path("cv_splits/train_cv_fold_0").is_file() is False:
            if stratified:
                skf = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)
                idx = 0
                for train_index, test_index in skf.split(self.df_train[[self.id_colname]], self.df_train[[self.target_colname]]):
                    self.df_train[[self.id_colname]].loc[train_index, :].to_csv('cv_splits/train_cv_fold_{}'.format(idx), index=False)
                    self.df_train[[self.id_colname]].loc[test_index, :].to_csv('cv_splits/test_cv_fold_{}'.format(idx), index=False)
                    idx += 1
            else:
                skf = KFold(n_splits=n_folds, random_state=42, shuffle=True)
                idx = 0
                for train_index, test_index in skf.split(self.df_train[[self.id_colname]]):
                    self.df_train[[self.id_colname]].loc[train_index, :].to_csv('cv_splits/train_cv_fold_{}'.format(idx), index=False)
                    self.df_train[[self.id_colname]].loc[test_index, :].to_csv('cv_splits/test_cv_fold_{}'.format(idx), index=False)
                    idx += 1
            gc.collect()

    def general_feature_engineering(self, train_only=True):
        """Feature engineering, which does NOT depend on train/test split"""
        if train_only:
            df = self.df_train
        else:
            print('Total Feature Eng',  time.ctime())    
            #df = pd.concat([self.df_train, self.df_test])
        if self.mode == 0:
            def basic_fe(df):
                df['EngineVersion_2'] = df['EngineVersion'].apply(lambda x: x.split('.')[2]).astype('category')
                df['EngineVersion_3'] = df['EngineVersion'].apply(lambda x: x.split('.')[3]).astype('category')

                df['AppVersion_1'] = df['AppVersion'].apply(lambda x: x.split('.')[1]).astype('category')
                df['AppVersion_2'] = df['AppVersion'].apply(lambda x: x.split('.')[2]).astype('category')
                df['AppVersion_3'] = df['AppVersion'].apply(lambda x: x.split('.')[3]).astype('category')

                df['AvSigVersion_0'] = df['AvSigVersion'].apply(lambda x: x.split('.')[0]).astype('category')
                df['AvSigVersion_1'] = df['AvSigVersion'].apply(lambda x: x.split('.')[1]).astype('category')
                df['AvSigVersion_2'] = df['AvSigVersion'].apply(lambda x: x.split('.')[2]).astype('category')

                df['OsBuildLab_0'] = df['OsBuildLab'].apply(lambda x: x.split('.')[0]).astype('category')
                df['OsBuildLab_1'] = df['OsBuildLab'].apply(lambda x: x.split('.')[1]).astype('category')
                df['OsBuildLab_2'] = df['OsBuildLab'].apply(lambda x: x.split('.')[2]).astype('category')
                df['OsBuildLab_3'] = df['OsBuildLab'].apply(lambda x: x.split('.')[3]).astype('category')

                df['Census_OSVersion_0'] = df['Census_OSVersion'].apply(lambda x: x.split('.')[0]).astype('category')
                df['Census_OSVersion_1'] = df['Census_OSVersion'].apply(lambda x: x.split('.')[1]).astype('category')
                df['Census_OSVersion_2'] = df['Census_OSVersion'].apply(lambda x: x.split('.')[2]).astype('category')
                df['Census_OSVersion_3'] = df['Census_OSVersion'].apply(lambda x: x.split('.')[3]).astype('category')

            
                df['primary_drive_c_ratio'] = df['Census_SystemVolumeTotalCapacity']/ df['Census_PrimaryDiskTotalCapacity']
                df['non_primary_drive_MB'] = df['Census_PrimaryDiskTotalCapacity'] - df['Census_SystemVolumeTotalCapacity']

                df['aspect_ratio'] = df['Census_InternalPrimaryDisplayResolutionVertical']/ df['Census_InternalPrimaryDisplayResolutionHorizontal']

                df['monitor_dims'] = df['Census_InternalPrimaryDisplayResolutionHorizontal'].astype(str) + '*' + df['Census_InternalPrimaryDisplayResolutionVertical'].astype('str')
                df['monitor_dims'] = df['monitor_dims'].astype('category')

                df['dpi'] = ((df['Census_InternalPrimaryDisplayResolutionHorizontal']**2 + df['Census_InternalPrimaryDisplayResolutionVertical']**2)**.5)/(df['Census_InternalPrimaryDiagonalDisplaySizeInInches'])

                df['dpi_square'] = df['dpi'] ** 2

                df['MegaPixels'] = (df['Census_InternalPrimaryDisplayResolutionHorizontal'] * df['Census_InternalPrimaryDisplayResolutionVertical'])/1e6

                df['Screen_Area'] = (df['aspect_ratio']* (df['Census_InternalPrimaryDiagonalDisplaySizeInInches']**2))/(df['aspect_ratio']**2 + 1)

                df['ram_per_processor'] = df['Census_TotalPhysicalRAM']/ df['Census_ProcessorCoreCount']

                df['new_num_0'] = df['Census_InternalPrimaryDiagonalDisplaySizeInInches'] / df['Census_ProcessorCoreCount']

                df['new_num_1'] = df['Census_ProcessorCoreCount'] * df['Census_InternalPrimaryDiagonalDisplaySizeInInches']
                
                df['Census_IsFlightingInternal'] = df['Census_IsFlightingInternal'].fillna(1)
                df['Census_ThresholdOptIn'] = df['Census_ThresholdOptIn'].fillna(1)
                df['Census_IsWIMBootEnabled'] = df['Census_IsWIMBootEnabled'].fillna(1)
                df['Wdft_IsGamer'] = df['Wdft_IsGamer'].fillna(0)

                df.SmartScreen = df.SmartScreen.str.lower()
                df.SmartScreen.replace({"promt":"prompt",
                                        "promprt":"prompt",
                                        "00000000":"0",
                                        "enabled":"on",
                                        "of":"off" ,
                                        "deny":"0" , # just one
                                        "requiredadmin":"requireadmin"
                                    },inplace=True)

                df.SmartScreen = df.SmartScreen.astype("category")

                def group_battery(x):
                    x = x.lower()
                    if 'li' in x: return 1
                    else:         return 0
                    
                df['isLithium_InternalBatteryType'] = df['Census_InternalBatteryType'].apply(group_battery) 

                add_cat_feats = [
                'Census_OSBuildRevision',
                'OsBuildLab',
                'SmartScreen',
                'AVProductsInstalled']
                for col1 in add_cat_feats:
                    for col2 in add_cat_feats:
                        if col1 != col2:
                            df[col1 + '__' + col2] = df[col1].astype(str) + df[col2].astype(str)
                            df[col1 + '__' + col2] = df[col1 + '__' + col2].astype('category')
                
                
                # 0 feature importance - TODO: further feature selection
                df = df.drop(['Census_PrimaryDiskTypeName',
                                #'GeoNameIdentifier',
                                'OsBuildLab_0',
                                #'CityIdentifier',
                                'OrganizationIdentifier',
                                'AvSigVersion_0',
                                'LocaleEnglishNameIdentifier',
                                'OsBuildLab_2',
                                'Platform',
                                'EngineVersion_3',
                                'OsVer',
                                'OsBuild',
                                #'OsSuite',
                                #'CountryIdentifier',
                                'Census_OSVersion_0',
                                'OsBuildLab_3',
                                'Census_IsTouchEnabled',
                                'Census_OSVersion_1',
                                'Census_OSVersion_2',
                                'HasTpm',
                                'DefaultBrowsersIdentifier',
                                #'aspect_ratio',
                                'IsSxsPassiveMode',
                                'dpi',
                                'dpi_square',
                                'MegaPixels',
                                'IsBeta',
                                'ram_per_processor', 
                                'new_num_0',
                                'Census_IsPenCapable',
                                'OsPlatformSubRelease',
                                #'Census_SystemVolumeTotalCapacity',
                                'UacLuaenable',
                                'Census_HasOpticalDiskDrive',
                                'Census_ProcessorClass',
                                'Census_ProcessorManufacturerIdentifier',
                                'Census_InternalPrimaryDisplayResolutionHorizontal',
                                'Census_InternalPrimaryDisplayResolutionVertical',
                                'Census_ProcessorCoreCount',
                                'Census_InternalBatteryType',
                                #'Census_InternalBatteryNumberOfCharges',
                                #'Census_OEMModelIdentifier',
                                'Census_OSBranch',
                                'Census_OSBuildNumber',
                                'Census_OSBuildRevision',
                                'Census_DeviceFamily',
                                'Firewall',
                                'Census_IsWIMBootEnabled',
                                #'IeVerIdentifier',
                                'PuaMode',
                                'Census_OSWUAutoUpdateOptionsName',
                                'Census_IsPortableOperatingSystem',
                                'Census_GenuineStateName',
                                'AutoSampleOptIn',
                                'Census_IsFlightingInternal',
                                'Census_IsFlightsDisabled',
                                'Census_FlightRing',
                                'Census_ThresholdOptIn',
                                'SkuEdition',
                                #'Census_FirmwareVersionIdentifier',
                                'Census_IsSecureBootEnabled', 
                                'SmartScreen__OsBuildLab', 'AVProductsInstalled__Census_OSBuildRevision', 'AVProductsInstalled__SmartScreen',
                                'SmartScreen__Census_OSBuildRevision', 'AVProductsInstalled__OsBuildLab',
                                'ProductName'], axis = 1)
                

                return df
            self.df_train, self.df_test = basic_fe(self.df_train), basic_fe(self.df_test)
            gc.collect()
            
            print('making bins',  time.ctime())
            '''
            #  making bins https://www.kaggle.com/guoday/nffm-baseline-0-690-on-lb
            def make_bucket(data,num=10):
                data.sort()
                bins=[]
                for i in range(num):
                    bins.append(data[int(len(data)*(i+1)//num)-1])
                return bins
            float_features=['Census_SystemVolumeTotalCapacity','Census_PrimaryDiskTotalCapacity']
            for f in float_features:
                self.df_train[f]=self.df_train[f].fillna(1e10)
                self.df_test[f]=self.df_test[f].fillna(1e10)
                data=list(self.df_train[f])+list(self.df_test[f])
                bins=make_bucket(data,num=50)
                self.df_train[f]=np.digitize(self.df_train[f],bins=bins)
                self.df_test[f]=np.digitize(self.df_test[f],bins=bins)
            del bins
            gc.collect()

            print('deleteting skewed vals',  time.ctime())
            for usecol in self.df_train.columns:
                if usecol in ['HasDetections', 'MachineIdentifier']:
                    continue
                self.df_train[usecol] = self.df_train[usecol].astype('str')
                self.df_test[usecol] = self.df_test[usecol].astype('str')
                
                #Fit LabelEncoder
                le = LabelEncoder().fit(
                        np.unique(self.df_train[usecol].unique().tolist()+
                                self.df_test[usecol].unique().tolist()))

                #At the end 0 will be used for dropped values
                self.df_train[usecol] = le.transform(self.df_train[usecol])+1
                self.df_test[usecol]  = le.transform(self.df_test[usecol])+1

                agg_tr = (self.df_train
                        .groupby([usecol])
                        .aggregate({'MachineIdentifier':'count'})
                        .reset_index()
                        .rename({'MachineIdentifier':'Train'}, axis=1))
                agg_te = (self.df_test
                        .groupby([usecol])
                        .aggregate({'MachineIdentifier':'count'})
                        .reset_index()
                        .rename({'MachineIdentifier':'Test'}, axis=1))

                agg = pd.merge(agg_tr, agg_te, on=usecol, how='outer').replace(np.nan, 0)
                #Select values with more than 1000 observations
                agg = agg[(agg['Train'] > 1000)].reset_index(drop=True)
                agg['Total'] = agg['Train'] + agg['Test']
                #Drop unbalanced values
                agg = agg[(agg['Train'] / agg['Total'] > 0.2) & (agg['Train'] / agg['Total'] < 0.8)]
                agg[usecol+'Copy'] = agg[usecol]

                self.df_train[usecol] = (pd.merge(self.df_train[[usecol]], 
                                        agg[[usecol, usecol+'Copy']], 
                                        on=usecol, how='left')[usecol+'Copy']
                                .replace(np.nan, 0).astype('int').astype('category'))

                self.df_test[usecol]  = (pd.merge(self.df_test[[usecol]], 
                                        agg[[usecol, usecol+'Copy']], 
                                        on=usecol, how='left')[usecol+'Copy']
                                .replace(np.nan, 0).astype('int').astype('category'))

                del le, agg_tr, agg_te, agg, usecol
                gc.collect()
            
            ## apply one hot encoding
            print('one hot encoding',  time.ctime())    
            cols_to_ohe = list(self.df_train.columns)
            cols_to_ohe.remove('HasDetections');
            cols_to_ohe.remove('MachineIdentifier');
            ohe = OneHotEncoder(categories='auto', sparse=False, dtype='uint8').fit(self.df_train[cols_to_ohe])
            self.df_train[cols_to_ohe] = ohe.transform(self.df_train[cols_to_ohe] )
            self.df_test [cols_to_ohe]  = ohe.transform(self.df_test[cols_to_ohe])
            '''
            gc.collect()
            
        elif self.mode == 1:
            pass
        else:
            raise ValueError('There is No Such Feature Engineering Mode')

    def _categorical_preprocess(self, df, cat_feature, how='ohe'):
        """Categorical variables preprocess"""
        assert how in ['ohe_encoder', 'label_encoder', 'target_encoder']

        if how == 'ohe_encoder':
            df.drop(cat_feature, 1, inplace=True)
            df = df.join(pd.get_dummies(df[cat_feature], prefix=cat_feature))

            # from sklearn.preprocessing import OneHotEncoder
            # ohe = OneHotEncoder(sparse=False)
            # y_ohe = ohe.fit_transform(y.values.reshape(-1, 1))
        elif how == 'label_encoder':
            le = LabelEncoder()
            df[cat_feature] = le.fit_transform(df[cat_feature])
        elif how == 'target_encoder':
            pass
        else:
            raise ValueError('There is no such categorical preprocessing')

        return df

    def fold_feature_engineering(self, train, test, total_test):
        """Feature engineering, which DOES depend on train/test split"""
        if self.mode == 0:

            print('Target en coding')
            to_encode =  ['AppVersion_1', 'AppVersion_2', 'AppVersion_3','AppVersion','SMode','Census_OSVersion', 'Census_OSVersion_3', 'OsBuildLab',
            'EngineVersion', 'EngineVersion_2', 'EngineVersion_3', 'GeoNameIdentifier','OsSuite',
             'AvSigVersion'  ,'OsPlatformSubRelease', 'OsPlatformSubRelease', 'SkuEdition', 'IeVerIdentifier'
            ,'AVProductStatesIdentifier'  
            ,'CityIdentifier'
            ,'CountryIdentifier'              
            ,'Census_OEMNameIdentifier'  
            ,'Census_OEMModelIdentifier'  
            ,'Census_ProcessorModelIdentifier'  
            ,'Census_InternalBatteryNumberOfCharges'  
            ,'Census_FirmwareVersionIdentifier'  
            ,'AvSigVersion_2' 
            ,'monitor_dims'  
            ,'Census_OSBuildRevision_OsBuildLab'  
            ,'Census_OSBuildRevision__SmartScreen'  
            ,'Census_OSBuildRevision__AVProductsInstalled'  
            ,'OsBuildLab__Census_OSBuildRevision'  
            ,'OsBuildLab__SmartScreen'  
            ,'OsBuildLab__AVProductsInstalled'  
            ,'SmartScreen__Census_OSBuildRevision'  
            ,'SmartScreen__OsBuildLab'  
            ,'AVProductsInstalled__Census_OSBuildRevision'  
            ,'AVProductsInstalled__OsBuildLab'  
            ]
            for encoding_variable in to_encode:
                if encoding_variable in train.columns:
                    print(encoding_variable, end = '; ')
                    te = TargetEncoding(10)
                    train, test, total_test = te.fit(train, test, total_test, encoding_variable, self.target_colname)
                else:
                    print('skipped', encoding_variable, end = '; ')
                gc.collect()
            print('Finished encoding')

            train = train[[col for col in train.columns if col not in [self.target_colname, self.id_colname]]]
            test = test [[col for col in test.columns  if col not in [self.target_colname, self.id_colname]]]
            total_test = total_test [[col for col in test.columns  if col not in [self.target_colname, self.id_colname]]]

            #test       =       test.astype(train.dtypes.to_dict())
            #total_test = total_test.astype(train.dtypes.to_dict())
            
        elif self.mode == 1:
            pass
        else:
            raise ValueError('There is No Such Feature Engineering Mode')
        return train, test, total_test



    def get_predictions(self, model_name, params, X_train, y_train, X_test, y_test):
        """Function to return predictions given data and model name"""
        cat_cols = [col for col in X_train.columns if col not in ['MachineIdentifier', 'Census_SystemVolumeTotalCapacity', 'HasDetections'] and str(X_train[col].dtype) == 'category']

        if model_name == 'xgboost':
            model = BesXGboost(params=params, metric=self.metric, maximize=self.maximize)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            # feat_imp = xgb.feature_importance()
            # print(feat_imp.head())

        elif model_name == 'lightgbm':
          
            train_data = lgb.Dataset(X_train, label=y_train, categorical_feature = cat_cols)
            valid_data = lgb.Dataset(X_test, label=y_test, categorical_feature = cat_cols)
            
            model = lgb.train(params,
                    train_data,
                    valid_sets = [train_data, valid_data],
                    verbose_eval=100, 
                    feval=eval_auc)

            del train_data, valid_data
            gc.collect()
            pred = model.predict(X_test, num_iteration=model.best_iteration)
        elif model_name == 'catboost':
            model = BesCatBoost(params=params, metric=self.metric.upper(), maximize=self.maximize)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

        elif model_name == 'logistic_regression':
            model = LogisticRegression()
            model.fit(X_train, y_train)
            pred = model.predict_proba(X_test)[:, 1]

        else:
            raise ValueError('There is No Such Model')

        return model, pred
    def plot_feature_importance (self, feature_importance):
        feature_importance["importance"] /= self.folds
        cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
            by="importance", ascending=False)[:10].index

        best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

        #plt.figure(figsize=(16, 12))
        sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
        #plt.title('LGB Features (avg over folds)')

    def run_single_model_validation_test_pred(self, model_name='lightgbm', params=None, oof_preds_path='', is_make_test_preds  = True, averaging = 'usual',
                                              is_plot_feature_importance = False):
        if oof_preds_path != '':
            oof_preds = pd.DataFrame()

        cv_metrics = [] 
        feature_importance = pd.DataFrame()
        prediction = np.zeros(len(self.df_test))
        test_preds = []

        for fold in range(self.folds):
            print('************************** FOLD {} **************************'.format(fold + 1),'\n\t', time.ctime())
            gc.collect()
            ids_train = pd.read_csv('cv_splits/train_cv_fold_{}'.format(fold))
            ids_test = pd.read_csv('cv_splits/test_cv_fold_{}'.format(fold))

            df_cv_train, df_cv_test = self.df_train.merge(ids_train), self.df_train.merge(ids_test)
            y_train, y_test = df_cv_train[self.target_colname], df_cv_test[self.target_colname]

            df_cv_train, df_cv_test, df_test = self.fold_feature_engineering(df_cv_train, df_cv_test, self.df_test)
            model, pred = self.get_predictions(model_name, params, df_cv_train, y_train, df_cv_test, y_test)

            if oof_preds_path != '':
                ids_test[os.path.split(oof_preds_path)[-1].split('.')[0]] = pred
                ids_test[self.target_colname] = y_test
                oof_preds = pd.concat([oof_preds, ids_test])

            met = self.compute_metric(y_test, pred)
            cv_metrics.append(met)
            print(met)
            if is_make_test_preds:
                y_pred = model.predict(df_test, num_iteration = model.best_iteration)
                test_preds.append(y_pred)

                del df_test
                gc.collect()
                if  averaging  == 'usual':
                    prediction += y_pred                    
                elif averaging == 'rank':
                    prediction += pd.Series(y_pred).rank().values # prediction /= prediction.max()                    
                #else:   prediction += (y_pred + pd.Series(y_pred).rank().values)/2 # TODO:  check if it`s possible
        
        if model_name == 'lightgbm':               
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = df_cv_train.columns
            fold_importance["importance"] = model.feature_importance()
            fold_importance["fold"] = fold + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)    

        #if oof_preds_path != '':
        #    oof_preds.to_csv(oof_preds_path, index=False)
        prediction /= self.folds
        del df_cv_train, df_cv_test, pred
        gc.collect()


        if model_name == 'lightgbm':
            if is_plot_feature_importance:
               self.plot_feature_importance(feature_importance)

        metric_mean = round(np.mean(cv_metrics), self.folds)
        metric_std = round(np.std(cv_metrics), self.folds)
        metric_overall = round(np.mean(cv_metrics) - np.std(cv_metrics), self.folds) if self.maximize else round(np.mean(cv_metrics) + np.std(cv_metrics), self.folds)
        print('{metric} mean: {mean}, {metric} std: {std}, {metric} overall: {ov}'.format(
            metric=self.metric,
            mean=metric_mean,
            std=metric_std,
            ov=metric_overall))
        print('ALL FOLDS:', [round(x, self.folds) for x in cv_metrics])

        #if is_make_test_preds:
        #    for_blending = {'train': oof_preds, 'test': test_preds}
            #joblib.dump(for_blending, oof_preds_path + model_name + '_auc_' + str(metric_mean) + '_std_' + str(metric_std) + '.pkl')
            
        gc.collect()
        return prediction, feature_importance

    def run_single_model_validation(self, model_name='xgboost', params=None, oof_preds_path=''):
        if oof_preds_path != '':
            oof_preds = pd.DataFrame()

        cv_metrics = []
        for fold in range(self.folds):
            print('************************** FOLD {} **************************'.format(fold + 1))
            gc.collect()
            ids_train = pd.read_csv('cv_splits/train_cv_fold_{}'.format(fold))
            ids_test = pd.read_csv('cv_splits/test_cv_fold_{}'.format(fold))

            df_cv_train, df_cv_test = self.df_train.merge(ids_train), self.df_train.merge(ids_test)
            y_train, y_test = df_cv_train[self.target_colname], df_cv_test[self.target_colname]

            df_cv_train, df_cv_test = self.fold_feature_engineering(df_cv_train, df_cv_test)
            model, pred = self.get_predictions(model_name, params, df_cv_train, y_train, df_cv_test)

            if oof_preds_path != '':
                ids_test[os.path.split(oof_preds_path)[-1].split('.')[0]] = pred
                ids_test[self.target_colname] = y_test
                oof_preds = pd.concat([oof_preds, ids_test])

            met = self.compute_metric(y_test, pred)
            cv_metrics.append(met)
            print(met)

        if oof_preds_path != '':
            oof_preds.to_csv(oof_preds_path, index=False)

        metric_mean = round(np.mean(cv_metrics), self.folds)
        metric_std = round(np.std(cv_metrics), self.folds)
        metric_overall = round(np.mean(cv_metrics) - np.std(cv_metrics), self.folds) if self.maximize else round(np.mean(cv_metrics) + np.std(cv_metrics), self.folds)
        print('{metric} mean: {mean}, {metric} std: {std}, {metric} overall: {ov}'.format(
            metric=self.metric,
            mean=metric_mean,
            std=metric_std,
            ov=metric_overall))
        print('ALL FOLDS:', [round(x, self.folds) for x in cv_metrics])
        return metric_mean, metric_std, metric_overall

    def run_stacked_model_validation(self, model_name='logistic_regression', params=None, prev_level_fold='oof_preds_level_1/', oof_preds_path=''):
        if oof_preds_path != '':
            oof_preds = pd.DataFrame()

        cv_metrics = []
        for fold in range(self.folds):
            print('************************** FOLD {} **************************'.format(fold + 1))
            ids_train = pd.read_csv('cv_splits/train_cv_fold_{}'.format(fold))
            ids_test = pd.read_csv('cv_splits/test_cv_fold_{}'.format(fold))

            df_train = pd.DataFrame()
            for f in os.listdir(prev_level_fold):
                path = os.path.join(prev_level_fold, f)
                if df_train.shape[0] == 0:
                    df_train = pd.read_csv(path)
                else:
                    df_train = df_train.merge(pd.read_csv(path))


            df_cv_train, df_cv_test = df_train.merge(ids_train), df_train.merge(ids_test)
            y_train, y_test = df_cv_train[self.target_colname], df_cv_test[self.target_colname]

            df_cv_train = df_cv_train[[col for col in df_cv_train.columns if col not in [self.target_colname, self.id_colname]]]
            df_cv_test = df_cv_test[[col for col in df_cv_test.columns if col not in [self.target_colname, self.id_colname]]]

            model, pred = self.get_predictions(model_name, params, df_cv_train, y_train, df_cv_test)

            if oof_preds_path != '':
                ids_test[os.path.split(oof_preds_path)[-1].split('.')[0]] = pred
                ids_test[self.target_colname] = y_test
                oof_preds = pd.concat([oof_preds, ids_test])

            met = self.compute_metric(y_test, pred)
            cv_metrics.append(met)
            print(met)

        if oof_preds_path != '':
            oof_preds.to_csv(oof_preds_path, index=False)

        metric_mean = round(np.mean(cv_metrics), self.folds)
        metric_std = round(np.std(cv_metrics), self.folds)
        metric_overall = round(np.mean(cv_metrics) - np.std(cv_metrics), self.folds) if self.maximize else round(np.mean(cv_metrics) + np.std(cv_metrics), self.folds)
        print('{metric} mean: {mean}, {metric} std: {std}, {metric} overall: {ov}'.format(
            metric=self.metric,
            mean=metric_mean,
            std=metric_std,
            ov=metric_overall))
        print('ALL FOLDS:', [round(x, self.folds) for x in cv_metrics])
        return metric_mean, metric_std, metric_overall

    def find_optimal_params(self, model_name='xgboost'):

        if model_name == 'xgboost':
            opt_params = BesXGboost.find_best_params(self)

        elif model_name == 'lightgbm':
            pass

        elif model_name == 'catboost':
            pass

        elif model_name == 'logistic_regression':
            pass

        else:
            raise ValueError('There is No Such Model')

        return opt_params

    def get_single_model_test_prediction(self, model_name='xgboost', params=None, preds_path=''):
        if preds_path == '':
            raise ValueError('Specify Path for Test Predictions')

        ids_test = self.df_test[[self.id_colname]] if self.id_colname is self.df_test.columns else self.df_test.reset_index()[['index']]

        y_train = self.df_train[self.target_colname]
        df_train, df_test = self.fold_feature_engineering(self.df_train, self.df_test)

        model, pred = self.get_predictions(model_name, params, df_train, y_train, df_test)


        ids_test[os.path.split(preds_path)[-1].split('.')[0]] = pred
        ids_test.to_csv(preds_path, index=False)

        return ids_test, model

    def get_stacked_model_test_prediction(self, model_name='logistic_regression', params=None,
                                          prev_level_test_fold = 'test_preds_level_1/', preds_path = ''):

        prev_level_train_fold = 'oof' + prev_level_test_fold[4:]
        df_train = pd.DataFrame()
        for f in os.listdir(prev_level_train_fold):
            path = os.path.join(prev_level_train_fold, f)
            if df_train.shape[0] == 0:
                df_train = pd.read_csv(path)
            else:
                df_train = df_train.merge(pd.read_csv(path))

        df_test = pd.DataFrame()
        for f in os.listdir(prev_level_test_fold):
            path = os.path.join(prev_level_test_fold, f)
            if df_test.shape[0] == 0:
                df_test = pd.read_csv(path)
            else:
                df_test = df_test.merge(pd.read_csv(path))

        ids_test = self.df_test[[self.id_colname]] if self.id_colname is self.df_test.columns else self.df_test.reset_index()[['index']]

        y_train = df_train[self.target_colname]

        df_train = df_train[
            [col for col in df_train.columns if col not in [self.target_colname, self.id_colname]]]
        df_test = df_test[
            [col for col in df_test.columns if col not in [self.target_colname, self.id_colname, 'index']]]

        model, pred = self.get_predictions(model_name, params, df_train, y_train, df_test)

        if preds_path == '':
            raise ValueError('Specify Path for Test Predictions')

        ids_test[os.path.split(preds_path)[-1].split('.')[0]] = pred
        ids_test.to_csv(preds_path, index=False)

        return ids_test, model