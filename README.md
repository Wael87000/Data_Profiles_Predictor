# Data_Profiles_Predictor
This project aims to help companies and HR departments to classify data profiles and resumes when applying to data related positions.  
The idea is to make sense of confusing data science job postings.  
Our use case is to predict the data related profiles from some information about the data specialist such as the experience, the technologies, the diploma, etc.   
The data specialist can be a data scientist, a data engineer, a data architect or a lead data scientist.    

Files:   
1- work.ipynb: data visualization, pre-processing, feature engineering, cross-validation and development of the following models: KNeighborsClassifier, Decision trees, Random Forest, AdaBoostClassifier, SVM  
2- work_catboost.ipynb: Pre-processing, feature engineering, cross-validation and development of the following model: Catboost   
3- deploy_catboost.py: deploy the web app with streamlit based on the catboost model aldready developed in the previous notebook   
4- track_catboost.py: tracking of the catboost model with mlflow
5- hyper_tunning_catboost.py: hyperparameters tunning of catboost with mlflow


