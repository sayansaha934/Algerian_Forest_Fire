import os
import pandas as pd
import pickle
import shutil
from data_preprocessing import preprocessing
from dBOperation.mongoDB import mongodbOperation
from application_logging.logger import App_Logger

class predictionClassification:
    '''
    This class will be used to predict output
    '''

    def __init__(self, data):
        self.data = data
        self.logging = App_Logger()

    def predictionFromModelClassification(self):
        try:
            if type(self.data) == dict:
                logging_db='classification_logging'
                logging_collection='single_data_logging'

                self.logging.log(logging_db, logging_collection, 'INFO', 'Prediction Started!!')
                self.logging.log(logging_db, logging_collection, 'INFO', f"user input: {self.data}")

                data = pd.DataFrame(self.data, index=[0])

                #Data Preprocessing
                self.logging.log(logging_db, logging_collection, 'INFO', 'Data Preprocessing started!!')
                preprocess = preprocessing.Preprocessor(logging_db, logging_collection)
                data = preprocess.addFeatures(data)
                data = preprocess.dropUnnecessaryColumns(data)
                data = preprocess.scaleNumericalValuesClassification(data)
                data = preprocess.encodeCategoricalValuesClassification(data)
                self.logging.log(logging_db, logging_collection, 'INFO', 'Successful End of Data Preprocessing!!')

                #Prediction From Model
                model = pickle.load(open('Models/Model_Classification.pickle', 'rb'))
                output = model.predict(data)[0]
                if output==1:
                    result='Fire'
                else:
                    result='Not Fire'

                self.logging.log(logging_db, logging_collection, 'INFO', f"Prediction output: {result}")
                self.logging.log(logging_db, logging_collection, 'INFO', 'Successful End of Prediction')

                return result
            else:
                logging_db = 'classification_logging'
                logging_collection = 'bulk_data_logging'

                self.logging.log(logging_db, logging_collection, 'INFO', 'Prediction Started!!')

                path=self.data

                #Database operation
                self.logging.log(logging_db, logging_collection, 'INFO', 'Database operation started!!')
                dbOps=mongodbOperation(logging_db, logging_collection)
                dbOps.insertIntoDatabase(path, 'classification_dataset')
                db_data_path=dbOps.extractDataFromDatabaseIntoCSV('classification_dataset')
                self.logging.log(logging_db, logging_collection, 'INFO', 'Successful End of Database operation!!')


                data=pd.read_csv(db_data_path)
                day_month_year=data[['day', 'month', 'year']]

                #Data Preprocessing
                self.logging.log(logging_db, logging_collection, 'INFO', 'Data Preprocessing started!!')
                preprocess = preprocessing.Preprocessor(logging_db, logging_collection)
                data = preprocess.addFeatures(data)
                data = preprocess.dropUnnecessaryColumns(data)
                data = preprocess.scaleNumericalValuesClassification(data)
                data = preprocess.encodeCategoricalValuesClassification(data)
                self.logging.log(logging_db, logging_collection, 'INFO', 'Successful End of Data Preprocessing!!')


                #Prediction From Model
                self.logging.log(logging_db, logging_collection, 'INFO', 'Prediction from model started!!')
                model = pickle.load(open('Models/Model_Classification.pickle', 'rb'))
                predictions = model.predict(data)
                output=[]
                for i in predictions:
                    if i == 1:
                        output.append('fire')
                    else:
                        output.append('not fire')

                output=pd.DataFrame(output, columns=['Classes'])
                final_output=pd.concat([day_month_year, output], axis=1)
                self.logging.log(logging_db, logging_collection, 'INFO', 'Successful End of Prediction from Model!!')


                #Preparation of Folder to send
                self.logging.log(logging_db, logging_collection, 'INFO', 'Started preparation of folder to send!!')
                folderName='Fire_Prtediction_Output'
                if not os.path.isdir(folderName):
                    os.mkdir(folderName)

                final_output.to_csv(folderName+"/"+"Fire_Prediction.csv", header=True, index=None)

                shutil.make_archive(folderName, 'zip', folderName)
                shutil.rmtree(folderName)
                os.remove(db_data_path)
                self.logging.log(logging_db, logging_collection, 'INFO', 'Successful End of Preparation of folder to send!!')
                self.logging.log(logging_db, logging_collection, 'INFO', 'Successful End of Prediction')

                return folderName+'.zip'

        except Exception as e:
            raise e




