
import os
from time import gmtime, strftime
import sagemaker
from sagemaker.tensorflow import TensorFlow
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

sagemaker_role = 'arn:aws:iam::70******AccountId:role/service-role/AmazonSageMaker-ExecutionRole'


def upload_training_data(session,base_job_name):
    traindata_s3_prefix = '{}/data/train'.format(base_job_name)
    train_s3 = session.upload_data(path='data/mnist.npz', key_prefix=traindata_s3_prefix)
    return train_s3


def sagemaker_estimator(sagemaker_role,code_entry,code_dir, instance_type, instance_count, hyperparameters, metric_definitions):
    sm_estimator = TensorFlow(entry_point=code_entry,
                              source_dir=code_dir,
                              role=sagemaker_role,
                              instance_type=instance_type,
                              instance_count=instance_count,
                              model_dir='/opt/ml/model',
                              hyperparameters=hyperparameters,
                              metric_definitions=metric_definitions,
                              framework_version='2.2',
                              py_version='py37',
                              script_mode=True)
    return sm_estimator


def sagemaker_training(sm_estimator,train_s3,training_job_name):
    sm_estimator.fit(train_s3, job_name=training_job_name, wait=False)


def sagemaker_hyperparam_tuning(sm_estimator, train_s3, hyperparameter_ranges, metric_definitions, tuning_job_name, max_jobs, max_parallel_jobs):
    objective_metric_name = 'validation:error'
    objective_type = 'Minimize'
    tuner = HyperparameterTuner(estimator=sm_estimator,
                                objective_metric_name=objective_metric_name,
                                hyperparameter_ranges=hyperparameter_ranges,
                                metric_definitions=metric_definitions,
                                max_jobs=max_jobs,
                                max_parallel_jobs=max_parallel_jobs,
                                objective_type=objective_type)

    tuner.fit(train_s3, job_name=tuning_job_name, wait=False)


if __name__ == '__main__':

    data_location = 'pycharm-training-tensorflow2'
    session = sagemaker.Session()
    bucket = session.default_bucket()

    train_s3 = upload_training_data(session, data_location)
    print(train_s3)

    code_entry = 'tf_script.py'
    code_dir = os.getcwd() + '/tf_code/'
    instance_type = 'ml.c5.xlarge'
    instance_count = 1
    hyperparameters = {'epochs': 10,
                       'batch_size': 128,
                       'learning_rate': 0.001,
                       'drop_rate': 0.8}
    metric_definitions = [
        {'Name': 'train:error', 'Regex': 'loss: ([0-9\\.]+)'},
        {'Name': 'validation:error', 'Regex': 'val_loss: ([0-9\\.]+)'},
        {'Name': 'validation:accuracy', 'Regex': 'val_accuracy: ([0-9\\.]+)'}
    ]
    sm_estimator = sagemaker_estimator(sagemaker_role, code_entry, code_dir, instance_type, instance_count, hyperparameters, metric_definitions)

    # sagemaker training job
    training_job_name = "tf-mnist-training-{}".format(strftime("%d-%H-%M-%S", gmtime()))
    sagemaker_training(sm_estimator, train_s3, training_job_name)

    # sagemaker tuning job
    hyperparameter_ranges = {
        'epochs': IntegerParameter(50, 200),
        'learning_rate': ContinuousParameter(0.0001, 0.1, scaling_type="Logarithmic"),
        'batch_size': IntegerParameter(32, 256),
        'drop_rate': ContinuousParameter(0.0, 1.0)
    }

    tuning_job_name = "tf-mnist-tuning-{}".format(strftime("%d-%H-%M-%S", gmtime()))
    max_jobs = 4
    max_parallel_jobs = 2
    #sagemaker_hyperparam_tuning(sm_estimator, train_s3, hyperparameter_ranges, metric_definitions, tuning_job_name, max_jobs, max_parallel_jobs)
