import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import sys
import dvc.api

path = 'data/wine-quality.csv'
repo = '/home/bens/git/data_versioning'
version ='v1'

data_url = dvc.api.get_url(path=path, repo=repo, rev=version)

print(data_url)
mlflow.set_experiment('demo')

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2
    
if __name__ == "__main__":
    np.random.seed(40)
    
    data = pd.read_csv(data_url, sep=";")
        
    train, test = train_test_split(data)
   
    train_x = train.drop(['quality'],axis=1)
    test_x = test.drop(['quality'], axis=1)
    train_y = train[['quality']]
    test_y = test[['quality']]
    
    #alpha = float(sys.argv[1]) if len(sys.argv) >1 else 0.5
    #l1_ratio = float(sys.argv[2]) if len(sys.argv) >2 else 0.5
    alpha = 0.4
    l1_ratio = 0.5
    
    
    #mlflow.sklearn.autolog()  #this is a all-in-one code...
    with mlflow.start_run() as run:
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)


        mlflow.log_param('data_url',data_url)
        mlflow.log_param('data_version',version)
        mlflow.log_param('input_rows', data.shape[0])
        mlflow.log_param('input_cols', data.shape[1])
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        from urllib.parse import urlparse
        #from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
        #from mlflow.tracking import MlflowClient
        # mlflow.set_tracking_uri("sqlite:///mlruns.db")
        # Register model name in the model registry
        #https://medium.com/datatau/how-to-setup-mlflow-in-production-a6f70511ebdc
        #
        #remote_server_uri = "postgresql+psycopg2://mlflow:mlflow123@localhost:3306/mlflow"
        #mlflow.set_tracking_uri(remote_server_uri)
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(lr, "model")
        
        print(tracking_url_type_store)
            
            
        #cols_x = pd.DataFrame(list(train_x.columns))
        #cols_x.to_csv('features.csv', header=False, index=False)
        #mlflow.log_artifact('features.csv')
    
        #cols_y = pd.DataFrame(list(train_y.columns))
        #cols_y.to_csv('targets.csv', header=False, index=False)
        #mlflow.log_artifact('targets.csv')
    
    # mlflow ui -p 10030 
    # mlflow server -p 10030 --backend-store-uri sqlite:///mlruns.db
    # mlflow server --backend-store-uri postgresql://mlflow:mlflow123@localhost/mlflow --default-artifact-root file:/home/bens/git/data_versioning/mlruns -h 0.0.0.0 -p 10030
    #

    

    
    
