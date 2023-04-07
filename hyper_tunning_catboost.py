# Import des libraries classique (numpy, pandas, ...)
import pandas as pd
import numpy as np
import re
import sklearn as sk
import seaborn as sb
from matplotlib import pyplot as plt

from sklearn.impute import SimpleImputer

import collections

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import learning_curve

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler

import catboost
from catboost import CatBoostClassifier, Pool

import mlflow

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


print("MLflow Tracking URI:", mlflow.get_tracking_uri())
print("MLflow Version:", mlflow.__version__)
print("CatBoost version:", catboost.__version__)
client = mlflow.tracking.MlflowClient()



imputerMedian = SimpleImputer(strategy='mean', missing_values=np.nan)
def imputationMedian(job, df):
    df.loc[df['Metier'] == job, 'Experience'] = imputerMedian.fit_transform(df.loc[df['Metier'] == job][['Experience']])
    return df


def build_data(data_path):
    data = pd.read_csv(data_path, delimiter=';')
    df = data.copy()

    df['Experience'] = [float(str(i).replace(",", ".")) for i in df['Experience']]
    df['Diplome'].replace(['Master','MSc','MASTER','master','msc','Mastere'],'Master', inplace=True)
    df['Diplome'].replace(['Bachelor','BSc','bachelor'],'Bachelor', inplace=True)
    df['Diplome'].replace(['PhD','phd','Phd'],'PhD', inplace=True)
    df['Diplome'].replace(['No diploma','NO'],'No diploma', inplace=True)

    for job in df['Metier'].unique(): 
        imputationMedian(job, df)

    df['Exp_label'] = df['Experience'].apply(lambda x: 'debutant' if x<2 else ('Confirme' if x>=2 and x<5 else ('Avance' if x>=5 and x<8 else 'Expert' )))

    df_new = df.copy()
    technos = df_new['Technologies'].str.split('/', expand=True)

    df_new[['technos1', 'technos2','technos3','technos4','technos5','technos6','technos7' ]] = df_new['Technologies'].str.split('/', expand=True)

    df_new = df_new.drop(['Technologies'], axis=1)
    technos_array = technos.stack().tolist()
    sorted_technos = sorted (technos_array, key=technos_array.count, reverse =True)

    output = []
    for x in sorted_technos:
        if x not in output:
            output.append(x)

    df_new = df_new.fillna(value= "No more")

    df_new.drop(['Entreprise', 'Experience'], axis = 1, inplace = True) 

    X_df = df_new.drop('Metier', axis = 1)
    y_df = df_new['Metier'] 

    X_train, X_val, y_train, y_val = train_test_split(X_df, y_df, test_size = 0.3)

    return X_train, X_val, y_train, y_val


def train(data_path, n_estimators, loss_function, learning_rate, depth, task_type, random_state, verbose, model_name):
    X_train, X_val, y_train, y_val = build_data(data_path)
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        print("MLflow:")
        print("  run_id:", run_id)
        print("  experiment_id:", experiment_id)
        print("  experiment_name:", client.get_experiment(experiment_id).name)

        # MLflow params
        print("Parameters:")
        print("  n_estimators:", n_estimators)
        print("  loss_function:", loss_function)
        print("  learning_rate:", learning_rate)
        print("  depth:", depth)
        print("  task_type:", task_type)
        print("  random_state:", random_state)
        print("  verbose:", verbose)


        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("loss_function", loss_function)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("depth", depth)
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("verbose", verbose)




        mlflow.set_tag("version.mlflow",mlflow.__version__)
        mlflow.set_tag("version.catboost",catboost.__version__)

        # Create and fit model
        cb = CatBoostClassifier(
                n_estimators=n_estimators,
                loss_function =loss_function,
                learning_rate = learning_rate,
                depth=depth,
                task_type = task_type,
                random_state = random_state,
                verbose = verbose) 
        

        pool_train = Pool(X_train, y_train, cat_features=['Diplome','Ville','Exp_label','technos1','technos2','technos3','technos4','technos5','technos6','technos7'])
        pool_val = Pool(X_val, cat_features=['Diplome','Ville','Exp_label','technos1','technos2','technos3','technos4','technos5','technos6','technos7'])

        cb_model = cb.fit(pool_train)

    
        print("model.type:", type(cb_model))
        print("model:", cb_model)

        # MLflow metrics
        y_pred = cb_model.predict(pool_val)
        print("predictions:",y_pred)

        accuracy = accuracy_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred, average='micro')
        precision = precision_score(y_val, y_pred, average='micro')
        f1 = f1_score(y_val, y_pred, average='micro')

        print("Metrics:")
        print("  accuracy:", accuracy)
        print("  recall:", recall)
        print("  precision:", precision)
        print("  f1:", f1)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1", f1)

        #Log model
        mlflow.xgboost.log_model(cb_model, "model", registered_model_name=model_name)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", dest="experiment_name", help="Experiment name", default="CatBoost-HyperOpt")
    parser.add_argument("--model_name", dest="model_name", help="Registered model name", default="Catboost_model")
    parser.add_argument("--data_path", dest="data_path", help="Data path", default='dataset_train_test.csv')
    parser.add_argument("--n_estimators", dest="n_estimators", help="n_estimators", default=200, type=int)
    parser.add_argument("--loss_function", dest="loss_function", help="loss function", default='MultiClass')#'MultiClass'
    parser.add_argument("--learning_rate", dest="learning_rate", help="learning rate", default=0.4)
    #parser.add_argument("--depth", dest="depth", help="depth", default=3)#3
    parser.add_argument("--task_type", dest="task_type", help="task type", default='CPU')
    parser.add_argument("--random_state", dest="random_state", help="random state", default=1)
    parser.add_argument("--verbose", dest="verbose", help="verbose", default=True)


    args = parser.parse_args()
    print("Arguments:")
    #for arg in vars(args):
        #print(f"  {arg}: {getattr(args, arg)}")
    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
    model_name = None if not args.model_name or args.model_name == "None" else args.model_name
    depth_list = [3,4,5,6]
    for depth in depth_list:
        train(args.data_path, args.n_estimators, args.loss_function, args.learning_rate, depth, args.task_type, args.random_state, args.verbose, model_name)
        


        


