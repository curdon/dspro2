import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import subprocess
import sys
# Ensure required package is installed
packages = ["azureml-core"]

for package in packages:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")

from azureml.core import Workspace, Dataset

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="name of registered dataset")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)
    parser.add_argument("--n_estimators", required=False, default=100, type=int)
    parser.add_argument("--registered_model_name", type=str, help="model name")
    args = parser.parse_args()
   
    # start Logging
    mlflow.start_run()

    # enable autologging
    mlflow.sklearn.autolog()

    ###################
    #<prepare the data>
    ###################
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)

    # load registered data
    subscription_id = '08539d0e-620d-424c-aea8-56853ed8fe3e'
    resource_group = 'dspro2'
    workspace_name = 'dspro2ml'

    workspace = Workspace(subscription_id, resource_group, workspace_name)

    dataset = Dataset.get_by_name(workspace, name=args.data)
    pddf_bh = dataset.to_pandas_dataframe()
    
    mlflow.log_metric("num_samples", pddf_bh.shape[0])
    mlflow.log_metric("num_features", pddf_bh.shape[1] - 1)

    # Your training code goes here
    target = "target"

    X = pddf_bh.drop(target, axis=1)
    y = pddf_bh[target]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_train_ratio, random_state=42)

    ###################
    #</prepare the data>
    ###################

    ##################
    #<train the model>
    ##################

    clf = RandomForestRegressor(n_estimators=args.n_estimators, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    ##################
    #</train the model>
    ##################

    ##########################
    #<save and register model>
    ##########################
    # registering the model to the workspace
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=clf,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name,
    )

    # saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=clf,
        path=os.path.join(args.registered_model_name, "trained_model"),
    )
    
    # stop Logging
    mlflow.end_run()

if __name__ == "__main__":
    main()