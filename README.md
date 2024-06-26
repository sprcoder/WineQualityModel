# Wine_Quality_Prediction
Using MLLib in Spark to train a ML model for wine quality prediction.

## Setup Information

* Start EMR instance in AWS with the following configurations
  * Provide the cluster name
  * emr-6.10.0 -> Spark 3.3.1 on Hadoop 3.3.3 YARN with and Zeppelin 0.10.1
  * 1 Primary instance and 3 Core Instance
    * m5.xlarge with minimum EBS storage
  * Enable -> Manually terminate cluster
  * Use the EC2 Key Pair already existing or create a new one
  * Amazon EMR service role(Default) -> EMR_DefaultRole
  * Instance Profile(Default) -> EMR_EC2_DefaultRole
* Create cluster with the above configurations.
* Connect to the cluster primary instance using ssh

### Environment setup in EC2 Primary Instance

* Do `aws configure` and store the aws credentials
* Install flintrock `pip install flintrock`
* add(export) path of flintrock to PATH variable
* Do `flintrock configure` and update the config.yml file with required configurations.
* Copy the Key Pair generated to the instance.
* Do `chmod 400 {keypair.pem}`
* Do `flintrock launch {cluster-name}` - Launches the flintrock with the environment specified in config file.
* Do `flintrock copy-file {cluster-name} {LocalFile} {RemoteDirectory}` - Copies the required files to the flintrock instance.
* Do `flintrock login {cluster-name}` - Logs in to the flintrock environment that has preinstalled spark and hadoop.

### Flintrock Environment Setup

* Do `aws configure` and store the aws credentials
* Install pyspark, boto3, pandas, scikit-learn -> `pip install {package}`
* Using the python file run the following command. Get master instance by running `flintrock describe ml-cluster` After launching flintrock.
```
spark-submit --deploy-mode client --master spark://{master-instance}:7077 wq_trainmodel.py
```
* Files available in github
  * `wq_trainmodel.py` - For taining the model using TrainingDataset.csv which is taken from s3 bucket.
  Note: Please create a directory `data` before executing the following file.
  * `wq_validatemodel.py` - Validates the model using ValidationDataset.csv which is taken from s3 bucket. Returns F1 score. 
  * `wq_testmodel.py` - Takes one argument and gives the f1 score or accuracy for the test csv data provided.
    * Argument can be "s3" if the file is taken from s3 bucket.
    * Argument can be csv file used to test.
  * `wq_testmodel_Local.py` - Can be implemented without hadoop file system.
    
## Wine Quality Prediction Model Summary

### Training the model

* TrainingDataset.csv is obtained from S3 bucket and a dataframe is created for it.
* Obtained the dataframe as features and lables to be used for model training using vector assembler
* The trained model is tested with validation dataset to fine tune the hyperparameters.
* Good results were shown for RandomForestClassifier with maxDepth=6, numTrees=30, impurity="gini"
* The trained model is saved and stored in s3 bucket.

### Validating the model

* ValidationDataset.csv is obtained from s3 bucket and a dataframe is created for it.
* The model to test the validation data frame is also taken from s3 bucket.
* Model is loaded as a RandomForestClassificationModel and tested with the validation data set.
* The result shows the F1 score of the validation data set.

#### Github link
https://github.com/sprcoder/Wine_Quality_Prediction

## Docker Implementation

* Created a flask application with integration of prediction model.
* Image is preloaded with trained model, hence it can be deployed anywhere and only the csv file is required to be uploaded as input.
* Dockerfile is placed at the root of the application with required commands to build the project.
* Docker image is created using the command `docker build -t mlapp .`
* Docker image is deployed using the command `docker run mlapp`
* The application can be accessed from the browser with the host address.
* Docker image is tagged to be uploaded to the dockerhub. `docker tag mlapp sr2484/sparkml_pa2:latest`
* Docker image is pushed to dockerhub with the command `docker push sr2484/sparkml_pa2:latest`

### Deploy docker image

* Image can be pulled with the command `docker pull sr2484/sparkml_pa2:latest`
* Run the docker run command `docker run -p 8000:80 sr2484/sparkml_pa2:latest`

#### DockerHub link
https://hub.docker.com/r/sr2484/sparkml_pa2/tags