# Flight Risk: Defining and Classifying Project Risk for NASA

Team: 
- Ainesh Pandey - ainesh93@gmail.com

## Abstract

<!-- For this challenge, we decided to assess the impact of changes in features that were measured across maternal hospital visits on different adverse pregnancy outcomes. To that extent, we divided the challenge's dataset into several components:
- a __covariates__ dataset with demographic and socio-economic information,
- a __deltas__ dataset capturing changes in features across multiple visits, and
- a __targets__ dataset with outcome variables related to maternal morbidity.

For each target variable, we trained and tuned 3 classification models: a _Logistic Regression Model_, a _Light Gradient Boosting Machine (LGBM)_ and a _Random Forest_. We assessed which of the three models perfomed the best in terms of the F-1 score (the harmonic mean between precision and recall metrics, going beyond the accuracy metric which we found to be unhelpful given the imbalanced distributions of the target variables) of the positive class. After dropping target variables with low support, we were able to identify impactful features for the following morbidities:
- Chronic Hypertension
- Postpartum Depression
- Postpartum Anxiety
- Preeclampsia

We then identified the 10 most important features in predicting each target and broke down the top features' univariate distributions by racial categories. We find that there are several impactful delta features related to the mother's sleep behavior, the mother's general health, pregnancy progression, and fetal health that also demonstrate distinctly different distributional behavior for the majority race class (white women) when compared to minority race classes. These findings can guide the direction of future research into the drivers of the APOs analyzed in this solution. -->

## White Paper

For a detailed explanation of the analysis performed for this challenge, please read the white paper (available in [DOCX]() and [PDF]() format).

## Running this project

In order to run this challenge submission, you will first need to run the `requirements.txt` file. After cloning the repo to your local drive, simply run `pip install -r requirements.txt` in your shell/terminal. This will adjust the package dependencies to match the ones used in our project. One of the models we used came from the LightGBM library, you may also want to navigate to the package's [documentation](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html) for additional steps should the requirements file return an error. You may need to run the `brew install libomp` command.

We have structured our analysis into 2 Jupyter notebooks. To reproduce our results, simply run through each notebook.

First, [GenerateRiskTarget.ipynb](https://github.com/ainesh1993/riskyspacebusiness/blob/main/notebooks/GenerateRiskTarget.ipynb) walks through the application of topic modeling using LDA to generate risk classifications. The categories of risk produced by our analysis are:
- `Risk Class 0`: Technical Execution Risk
- `Risk Class 1`: Managerial Process Risk
- `Risk Class 2`: Operational Cost Risk

Second, [ModelRiskClass.ipynb](https://github.com/ainesh1993/riskyspacebusiness/blob/main/notebooks/ModelRiskClass.ipynb) runs a gamut of multi-class models on provided project metadata to predict the probabilty of certain risks for NASA projects. The best ensemble model developed, along with a series of base classifiers and pre-processing vectorizers, are saved in the models directory.

Furthermore, the `classify_risk.py` script has been developed and saved in the `scripts` directory of the project (within the `notebooks` directory) to facilitate two processes for NASA in the future:
- the batch classification of multiple projects using a CSV file (the [ClassifyRisk_batch.ipynb](https://github.com/ainesh1993/riskyspacebusiness/blob/main/notebooks/ClassifyRisk_batch.ipynb) notebook serves as an example of this process)
- the classification of an individual project using its *Title* and *Abstract* (the [ClassifyRisk.ipynb](https://github.com/ainesh1993/riskyspacebusiness/blob/main/notebooks/ClassifyRisk.ipynb) notebook serves as an example of this process).

Simply update the user inputs (which are heavily documented for ease of use) defined at the beginning of these notebooks to classify future NASA projects.