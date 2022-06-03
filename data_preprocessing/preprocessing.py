import pandas as pd
import pickle
import sklearn
from application_logging.logger import App_Logger

class Preprocessor:
    '''
    This class will be used to preprocess the data before prediction
    '''

    def __init__(self, logging_db, logging_collection):
        self.logging_db = logging_db
        self.logging_collection = logging_collection
        self.logging = App_Logger()

    def addFeatures(self, data):
        '''
            Method Name: addFeatures
            Description: It adds day_name feature and converts month column
                        to month name
            Output: A Dataframe
            On  Failure: Raise Exception

            Written by: Sayan Saha
            Version: 1.0
            Revision: None
        '''

        try:
            data['day_name'] = pd.to_datetime(data[['day', 'month', 'year']]).apply(lambda x: x.strftime("%a"))
            data['month'] = pd.to_datetime(data[['day', 'month', 'year']]).apply(lambda x: x.strftime("%b"))
            self.logging.log(self.logging_db, self.logging_collection, 'INFO', 'Added Features Successfully!!')
            return data
        except Exception as e:
            self.logging.log(self.logging_db, self.logging_collection, 'ERROR', F"Error occured to add features: {e}")
            raise e

    def dropUnnecessaryColumns(self, data):
        '''
                    Method Name: dropUnnecessaryColumns
                    Description: It drops year, day, BUI, FWI columns
                    Output: A Dataframe without year, day, BUI, FWI columns
                    On Failure: Raise Exception

                    Written by: Sayan Saha
                    Version: 1.0
                    Revision: None
                '''
        try:
            data = data.drop(columns=['year', 'day', 'BUI', 'FWI'])
            self.logging.log(self.logging_db, self.logging_collection, 'INFO', 'Dropped Unnecessary Columns Successfully!!')
            return data
        except Exception as e:
            self.logging.log(self.logging_db, self.logging_collection, 'ERROR', f"Error occured to drop columns: {e}")
            raise e

    def scaleNumericalValuesClassification(self, data):
        '''
                            Method Name: scaleNumericalValuesClassification
                            Description: It scales numerical values
                            Output: A Dataframe with scaled numerical values
                            On Failure: Raise Exception

                            Written by: Sayan Saha
                            Version: 1.0
                            Revision: None
                '''

        try:
            num_df = data.drop(columns=['month', 'day_name', 'Region'])
            cat_df = data[['month', 'day_name', 'Region']]
            scaler = pickle.load(open('Scaling/Scaler_Classification.pickle', 'rb'))
            num_array = scaler.transform(num_df)
            num_df = pd.DataFrame(num_array, columns=num_df.columns)
            final_data = pd.concat([num_df, cat_df], axis=1)
            self.logging.log(self.logging_db, self.logging_collection, 'INFO', 'Scaled Numerical Values Successfully!!')
            return final_data

        except Exception as e:
            self.logging.log(self.logging_db, self.logging_collection, 'ERROR', f"Error occured to Scale Numerical Values: {e}")
            raise e

    def encodeCategoricalValuesClassification(self, data):
        '''
                            Method Name: encodeCategoricalValuesClassification
                            Description: It encodes categorical values
                            Output: A Dataframe with encoded categorical values
                            On Failure: Raise Exception

                            Written by: Sayan Saha
                            Version: 1.0
                            Revision: None
                        '''

        try:
            data['month'] = data['month'].map({'Aug': 3, 'Jul': 2, 'Jun': 1, 'Sep': 0})
            data['Region'] = data['Region'].map({'SBAR': 1, 'BR': 0})
            encoder = pickle.load(open('Encoder/encode_dayname.pickle', 'rb'))
            day_names_array = encoder.transform(data[['day_name']])
            day_names = pd.DataFrame(day_names_array, columns=encoder.get_feature_names_out())
            data = data.drop(columns=['day_name'])
            data = pd.concat([data, day_names], axis=1)
            self.logging.log(self.logging_db, self.logging_collection, 'INFO', 'Encoded Categorical Values Successfully!!')

            return data

        except Exception as e:
            self.logging.log(self.logging_db, self.logging_collection, 'ERROR', f"Error occured to encode Categorical Values: {e}")

            raise e

    def scaleNumericalValuesRegression(self, data):
        '''
                            Method Name: scaleNumericalValuesRegression
                            Description: It scales numerical values
                            Output: A Dataframe with scaled numerical values
                            On Failure: Raise Exception

                            Written by: Sayan Saha
                            Version: 1.0
                            Revision: None
                        '''

        try:
            num_df = data.drop(columns=['month', 'day_name', 'Classes', 'Region'])
            cat_df = data[['month', 'day_name', 'Classes', 'Region']]
            scaler = pickle.load(open('Scaling/Scaler_Regression.pickle', 'rb'))
            num_array = scaler.transform(num_df)
            num_df = pd.DataFrame(num_array, columns=num_df.columns)
            final_data = pd.concat([num_df, cat_df], axis=1)
            self.logging.log(self.logging_db, self.logging_collection, 'INFO', 'Scaled Numerical Values Successfully!!')
            return final_data

        except Exception as e:
            self.logging.log(self.logging_db, self.logging_collection, 'ERROR', f"Error occured to Scale Numerical Values: {e}")
            raise e



    def encodeCategoricalValuesRegression(self, data):
        '''
                            Method Name: encodeCategoricalValuesRegression
                            Description: It encodes categorical values
                            Output: A Dataframe with encoded categorical values
                            On Failure: Raise Exception

                            Written by: Sayan Saha
                            Version: 1.0
                            Revision: None
                        '''

        try:
            data['month'] = data['month'].map({'Aug': 3, 'Jul': 2, 'Jun': 1, 'Sep': 0})
            data['Region'] = data['Region'].map({'SBAR': 1, 'BR': 0})
            data['Classes'] = data['Classes'].map({'fire': 1, 'not fire': 0})
            encoder = pickle.load(open('Encoder/encode_dayname.pickle', 'rb'))
            day_names_array = encoder.transform(data[['day_name']])
            day_names = pd.DataFrame(day_names_array, columns=encoder.get_feature_names_out())
            data = data.drop(columns=['day_name'])
            data = pd.concat([data, day_names], axis=1)
            self.logging.log(self.logging_db, self.logging_collection, 'INFO', 'Encoded Categorical Values Successfully!!')

            return data

        except Exception as e:
            self.logging.log(self.logging_db, self.logging_collection, 'ERROR', f"Error occured to encode Categorical Values: {e}")

            raise e
