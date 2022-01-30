import pandas as pd
import numpy as np
import statistics
import pickle

class ClassifyRisk:
    """
    This package can be used by NASA to classify the risk of certain projects, as determined by our analyses. As a quick
    reminder, the risk classes are as follows:
        - `Risk Class 0`: Catastrophic Failure
        - `Risk Class 1`: High-Cost Aviation Projects
        - `Risk Class 2`: XYZ
        - `Risk Class 4`: Catch-All

    NASA can provide a dataframe that contains at least the Title and Abstract of the project.
    """
    
    def __init__(self, df: pd.DataFrame=None):
        """
        Initialize ClassifyRisk object by importing all of the required vectorizers and models, preprocess the data
        provided by NASA

        Inputs:
            - df: the dataframe provided by NASA
                - Type: pd.DataFrame
                - Default value: None

        Returns: N/A
        """
        # open the vectorizers created in the ModelRiskClass notebook
        with open('../models/tfidf_vectorizer.pkl', 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        with open('../models/pca_vectorizer.pkl', 'rb') as f:
            self.pca_vectorizer = pickle.load(f)

        # open the models created in the ModelRiskClass notebook
        with open('../models/model_lm.pkl', 'rb') as f:
            self.model_lm = pickle.load(f)
        with open('../models/model_rfc.pkl', 'rb') as f:
            self.model_rfc = pickle.load(f)
        with open('../models/model_lgb.pkl', 'rb') as f:
            self.model_lgb = pickle.load(f)
        with open('../models/model_gnb.pkl', 'rb') as f:
            self.model_gnb = pickle.load(f)
        
        # access the dataframe provided by NASA
        self.df = df

    def preprocess_and_classify(self, title_col: str='Title', abstract_col: str='Abstract'):
        """
        Preprocess the dataframe provided by NASA, and classifies the projects in the dataframe by the Risk Class

        Inputs:
        - title_col: the header of the 'Title' column, in case it has changed
                - Type: str
                - Default value: 'Title'
            - abstract_col: the header of the 'Abstract' column, in case it has changed
                - Type: str
                - Default value: 'Abstract'
        
        Returns: N/A
        """
        # keep only the title and abstract columns
        temp_df = self.df[[title_col, abstract_col]]

        # generate 'Description' by appending title and abstract
        temp_df['Description'] = temp_df[title_col].astype(str) + ' ' + temp_df[abstract_col].astype(str)
        temp_df.drop([title_col, abstract_col], axis=1, inplace=True)

        # call the create_pca_representation() function to generate the PCA matrix from the description
        pca_matrix_pro = self.create_pca_representation(temp_df)
        temp_df.drop('Description', axis=1, inplace=True)
        temp_df = pd.concat([self.df, pd.DataFrame(pca_matrix_pro)], axis=1)

        # create predictions on input data using all of the models from the ModelRiskClass notebook
        pred_lm  = self.model_lm.predict(self.df)
        pred_rfc = self.model_rfc.predict(self.df)
        pred_lgb = self.model_lgb.predict(self.df)
        pred_gnb = self.model_gnb.predict(self.df)

        # ensemble the predictions
        df_pred = pd.DataFrame([pred_lm, pred_rfc, pred_lgb, pred_gnb]).T
        pred = df_pred.apply(statistics.mode, axis=1)

        # convert predictions into usable format
        pred = pred.replace({0: 'Catastrophic Failure',
                             1: 'High-Cost Aviation Projects',
                             2: 'XYZ',
                             4: 'Catch-All'})
        self.df['Risk Class'] = pred
        
        # return dataframe with risk identified
        return self.df

    def create_pca_representation(self):
        """
        Modified version of create_pca_representation() from the ModelRiskClass notebook

        Inputs: N/A
        Returns:
            - pca_matrix: transformed PCA representation of the Description column
        """
        # transform the Description column using the tfidf_vectorizer
        tfidf_matrix = self.tfidf_vectorizer.transform(self.df['Description'])
        tfidf_df     = pd.DataFrame(tfidf_matrix.todense())

        # return the tfidf matrix transformed by the pca_vectorizer
        return self.pca_vectorizer.transform(tfidf_df)