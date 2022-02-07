# Flight Risk: Defining and Classifying Project Risk for NASA

Team: 
- Ainesh Pandey - ainesh93@gmail.com

## Abstract

The National Aeronautics and Space Administration (NASA) executes complicated projects prone to a variety of risk. To alleviate these risks, NASA wants to develop an AI/ML solution to categorize and predict risk for future projects. In this white paper, we address this problem by applying topic modeling with LDA to extract risk categories and training a gamut of multi-class modeling algorithms to predict future risk.

Topic modeling with LDA revealed three main categories of risk: technical execution risk, managerial process risk, and operational cost risk. After classifying the risk of each past project, we trained and tuned base multi-class models and developed custom ensembles. Our final model, an ensemble of three base classifiers, performs with a stable 79% accuracy, macro-average F1, and weighted-average F1 with just the projects’ title and abstract as inputs. Using this model, NASA can quickly and accurately predict potential pitfalls in future projects and adjust their execution accordingly, increasing future project success rate and leading to fewer adverse outcomes.

## White Paper

For a detailed explanation of the analysis performed for this challenge, please read the white paper (available in [DOCX](https://github.com/ainesh1993/riskyspacebusiness/blob/main/Flight%20Risk%20-%20White%20Paper.docx) and [PDF](https://github.com/ainesh1993/riskyspacebusiness/blob/main/Flight%20Risk%20-%20White%20Paper.pdf) format).

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