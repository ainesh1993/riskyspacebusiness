import pandas as pd
import pickle
from scripts.utils import preprocess_string_pro, preprocess_string_aug

import warnings
warnings.filterwarnings("ignore")

class ClassifyRisk:
    """
    This package can be used by NASA to classify the risk of certain projects, as determined by our analyses. As a quick
    reminder, the risk classes are as follows:
        - `Risk Class 0`: Technical Execution Risk
        - `Risk Class 1`: Managerial Process Risk
        - `Risk Class 2`: Operational Cost Risk

    NASA can provide a dataframe that contains at least the Title and Abstract of each project for batch classification,
    or they can provide an individual Title and Abstract for one project to classify.
    """
    
    def __init__(self, df: pd.DataFrame=None, model_type: str='ensemble'):
        """
        Initialize ClassifyRisk object by importing all of the required vectorizers and models

        Inputs:
            - df: the dataframe provided by NASA, if any
                - Type: pd.DataFrame
                - Default value: None
            - model_type: the type of model NASA wants to use
                - Type: str
                - Default value: 'ensemble'

        Returns: N/A
        """
        # open the vectorizers created in the ModelRiskClass notebook
        with open('../models/tfidf_vectorizer.pkl', 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        with open('../models/pca_vectorizer.pkl', 'rb') as f:
            self.pca_vectorizer = pickle.load(f)

        # identify the model filename specified by user input
        model_filename = None
        if   model_type=='ensemble': model_filename = '../models/model_ensemble.pkl'
        elif model_type=='lm':       model_filename = '../models/model_lm.pkl'
        elif model_type=='rfc':      model_filename = '../models/model_rfc.pkl'
        elif model_type=='lgb':      model_filename = '../models/model_lgb.pkl'
        elif model_type=='knn':      model_filename = '../models/model_knn.pkl'
        elif model_type=='gnb':      model_filename = '../models/model_gnb.pkl'
        else:                        model_filename = '../models/model_ensemble.pkl'

        # open the model specified by user input
        with open(model_filename, 'rb') as f:
            self.model = pickle.load(f)

        # access the dataframe provided by user input
        self.df = df

    def classify(self, title: str='', abstract: str=''):
        """
        Preprocess the Title and Abstract provided by NASA, and classify the project

        Inputs:
            - title: the title of the project
                - Type: str
                - Default value: ''
            - abstract_col: the abstract of the project
                - Type: str
                - Default value: ''
        
        Returns:
            - dict: dictionary with classification details for the provided project
        """
        # generate 'Description' by appending title and abstract
        description = title + ' ' + abstract
        temp_df = pd.DataFrame([description], columns=['Description'])

        # call the create_pca_representation() function to generate the PCA matrix from the description
        pca_representation = self.create_pca_representation(temp_df['Description'])
        temp_df.drop('Description', axis=1, inplace=True)
        temp_df = pd.concat([temp_df, pd.DataFrame(pca_representation)], axis=1)

        # get probabilities of each risk class using the ensemble model from the ModelRiskClass notebook
        risk_probs = self.model.predict_proba(temp_df)

        # create risk profile from probabilities
        risk_profile = {}
        codes = range(0, 3)
        for code in codes:
            desc = None
            if   code==0: desc = 'Technical Execution Risk'
            elif code==1: desc = 'Managerial Process Risk'
            elif code==2: desc = 'Operational Cost Risk'

            prob = risk_probs[0][code]
            risk_profile[desc] = prob

        # create predictions on input data using the ensemble model
        highest_code = pd.Series(self.model.predict(temp_df))
        highest_prob = [max(x) for x in risk_probs]

        # convert predictions into understandable format
        highest_desc = highest_code.replace({0: 'Technical Execution Risk',
                                             1: 'Managerial Process Risk',
                                             2: 'Operational Cost Risk'})
        
        # return dataframe with highest risk identified and risk profile
        return {'Highest Risk Class Code': highest_code[0],
                'Highest Risk Class': highest_desc[0],
                'Highest Risk Class Probability': highest_prob[0],
                'Overall Risk Profile': risk_profile}

    def batch_classify(self, id_col: str='Lesson ID', title_col: str='Title', abstract_col: str='Abstract'):
        """
        Preprocess the dataframe provided by NASA, and classify the projects

        Inputs:
            - id_col: the header of the project identifier column, in case it has changed
                - Type: str
                - Default value: 'Lesson ID'
            - title_col: the header of the 'Title' column, in case it has changed
                - Type: str
                - Default value: 'Title'
            - abstract_col: the header of the 'Abstract' column, in case it has changed
                - Type: str
                - Default value: 'Abstract'
        
        Returns:
            - df: dataframe containing predictions and probabilities for all projects
        """
        # keep only the title and abstract columns
        temp_df = self.df[[title_col, abstract_col]]

        # generate 'Description' by appending title and abstract
        temp_df['Description'] = temp_df[title_col].astype(str) + ' ' + temp_df[abstract_col].astype(str)
        temp_df.drop([title_col, abstract_col], axis=1, inplace=True)

        # call the create_pca_representation() function to generate the PCA matrix from the description
        pca_matrix_pro = self.create_pca_representation(temp_df['Description'])
        temp_df.drop('Description', axis=1, inplace=True)
        temp_df = pd.concat([temp_df, pd.DataFrame(pca_matrix_pro)], axis=1)

        # get probabilities of each risk class using the ensemble model from the ModelRiskClass notebook
        risk_probs = self.model.predict_proba(temp_df)

        # create predictions on input data using the ensemble model
        pred_code = pd.Series(self.model.predict(temp_df))
        pred_prob = [max(x) for x in self.model.predict_proba(temp_df)]

        # convert predictions into understandable format
        pred_desc = pred_code.replace({0: 'Technical Execution Risk',
                                       1: 'Managerial Process Risk',
                                       2: 'Operational Cost Risk'})

        self.df['Highest Risk Class Code'] = pred_code
        self.df['Highest Risk Class'] = pred_desc
        self.df['Highest Risk Class Probability'] = pred_prob
        self.df['Technical Execution Risk'] = [x[0] for x in risk_probs]
        self.df['Managerial Process Risk'] = [x[1] for x in risk_probs]
        self.df['Operational Cost Risk'] = [x[2] for x in risk_probs]
        
        # return dataframe with risk identified
        return self.df[[id_col, title_col, abstract_col,
                        'Highest Risk Class Code', 'Highest Risk Class', 'Highest Risk Class Probability',
                        'Technical Execution Risk', 'Managerial Process Risk', 'Operational Cost Risk']]

    def create_pca_representation(self, desc: pd.Series=None):
        """
        Modified version of create_pca_representation() from the ModelRiskClass notebook

        Inputs:
            - desc: column containing description from dataframe
                - Type: pd.Series
                - Default value: None

        Returns:
            - pca_matrix: transformed PCA representation of the Description column
        """
        # transform the Description column using the tfidf_vectorizer
        tfidf_matrix = self.tfidf_vectorizer.transform(desc)
        tfidf_df     = pd.DataFrame(tfidf_matrix.todense())

        # return the tfidf matrix transformed by the pca_vectorizer
        return self.pca_vectorizer.transform(tfidf_df)