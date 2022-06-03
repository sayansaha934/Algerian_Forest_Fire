import os.path
import pymongo
import pandas as pd
from application_logging.logger import App_Logger

class mongodbOperation:

    def __init__(self, logging_db, logging_collection):
        self.client = pymongo.MongoClient(
            "mongodb+srv://fire:fire@algerian-forest-fire.tkzxx.mongodb.net/?retryWrites=true&w=majority")
        self.logging_db=logging_db
        self.logging_collection=logging_collection
        self.logging=App_Logger()

    def insertIntoDatabase(self, path, collection_name):
        '''
                    Method Name: insertIntoDatabase
                    Description: It inserts data into database
                    Output: None
                    On  Failure: Raise Exception

                    Written by: Sayan Saha
                    Version: 1.0
                    Revision: None
                '''
        try:
            database_name = 'Prediction_dataset'
            database = self.client[database_name]
            collection = database[collection_name]

            if database_name in self.client.list_database_names():
                if collection_name in database.list_collection_names():
                    collection.drop()
                    collection = database[collection_name]
            df = pd.read_csv(path)
            for i, row in df.iterrows():
                collection.insert_one(dict(row))
            self.logging.log(self.logging_db, self.logging_collection, 'INFO', 'File inserted into Database Successfully!!')

        except Exception as e:
            self.logging.log(self.logging_db, self.logging_collection, 'ERROR', f"Error occured to insert data into database: {e}")
            raise e

    def extractDataFromDatabaseIntoCSV(self, collection_name):
        '''
                            Method Name: extractDataFromDatabaseIntoCSV
                            Description: It extracts data from database and stores it in a csv file
                            Output: Path of the csv file
                            On  Failure: Raise Exception

                            Written by: Sayan Saha
                            Version: 1.0
                            Revision: None
                        '''
        try:
            if not os.path.isdir('Prediction_FileFromDB'):
                os.mkdir('Prediction_FileFromDB')

            database_name = 'Prediction_dataset'
            database = self.client[database_name]
            collection = database[collection_name]

            df = pd.DataFrame(collection.find({}, {'_id': 0}))
            path = 'Prediction_FileFromDB/'+collection_name+'.csv'
            df.to_csv(path, header=True, index=None)
            self.logging.log(self.logging_db, self.logging_collection, 'INFO', 'Extracted data from Database successfully!!')

            return path
        except Exception as e:
            self.logging.log(self.logging_db, self.logging_collection, 'ERROR', f"Error occured to extract data from database: {e}")
            raise e