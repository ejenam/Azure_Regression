
from matplotlib import pyplot as plt
import numpy as np
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import pandas as pd
import mlflow
import mlflow.sklearn

def select_first_file(path):
    """Selects first file in folder, use under assumption there is only one file in folder
    Args:
        path (str): path to directory or file to choose
    Returns:
        str: full path of selected file
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])


# Start Logging
mlflow.start_run()

# enable autologging
mlflow.sklearn.autolog()

os.makedirs("./outputs", exist_ok=True)

TARGET_COL = "cost"

NUMERIC_COLS = [
    "distance", "dropoff_latitude", "dropoff_longitude", "passengers", "pickup_latitude",
    "pickup_longitude", "pickup_weekday", "pickup_month", "pickup_monthday", "pickup_hour",
    "pickup_minute", "pickup_second", "dropoff_weekday", "dropoff_month", "dropoff_monthday",
    "dropoff_hour", "dropoff_minute", "dropoff_second"
]

CAT_NOM_COLS = [
    "store_forward", "vendor"
]

CAT_ORD_COLS = [
]
    

def main():
    """Main function of the script."""
    
    
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data', type=str)
    parser.add_argument("--n_estimators", dest='n_estimators', type=int, default=500)
    parser.add_argument("--max_depth", dest='max_depth', type=int, default=1)
    parser.add_argument('--max_features', type=str, default='auto',
                        help='Number of features to consider at every split')
    parser.add_argument("--min_samples_leaf", dest='min_samples_leaf', type=int, default=4)
    parser.add_argument("--min_samples_split", dest='min_samples_split', type=int, default=5)
    parser.add_argument("--registered_model_name", type=str, help="model name")
    parser.add_argument("--model", type=str, help="path to model file")
    # parse args
    args = parser.parse_args()
    
    # paths are mounted as folder, therefore, we are selecting the file from folder
    train_data = pd.read_csv(select_first_file(args.training_data))

    
    # Split the data into input(X) and output(y)
    y_train = train_data[TARGET_COL]
    X_train = train_data[NUMERIC_COLS + CAT_NOM_COLS + CAT_ORD_COLS]
    
    
    # Train a Random Forest Regression Model with the training set
    model = RandomForestRegressor(n_estimators = args.n_estimators,
                                  max_depth = args.max_depth,
                                  max_features = args.max_features,
                                  min_samples_leaf = args.min_samples_leaf,
                                  min_samples_split = args.min_samples_split,
                                  random_state=0)
    
    
    # log model hyperparameters
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("max_features", args.max_features)
    mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
    mlflow.log_param("min_samples_split", args.min_samples_split)
    

    # Train model with the train set
    model.fit(X_train, y_train)

    # Predict using the Regression Model
    yhat_train = model.predict(X_train)

    # Evaluate Regression performance with the train set
    r2 = r2_score(y_train, yhat_train)
    mse = mean_squared_error(y_train, yhat_train)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train, yhat_train)
    
    # log model performance metrics
    mlflow.log_metric("train r2", r2)
    mlflow.log_metric("train mse", mse)
    mlflow.log_metric("train rmse", rmse)
    mlflow.log_metric("train mae", mae)

    # Visualize results
    plt.scatter(y_train, yhat_train,  color='black')
    plt.plot(y_train, y_train, color='blue', linewidth=3)
    plt.xlabel("Real value")
    plt.ylabel("Predicted value")
    plt.savefig("regression_results.png")
    mlflow.log_artifact("regression_results.png")
    
    # Registering the model to the workspace
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name,
    )

    # Saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=model,
        path=os.path.join(args.model, "trained_model"),
    )
    
    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()
