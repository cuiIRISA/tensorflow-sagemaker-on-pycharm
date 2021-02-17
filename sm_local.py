from tensorflow.keras import datasets
import numpy as np
from sagemaker.tensorflow import TensorFlow
import os

from sagemaker.local import LocalSession

sagemaker_role = 'arn:aws:iam::70******AccountId:role/RoleNameHere'
sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}


def sagemaker_estimator(sagemaker_role, code_entry, code_dir, hyperparameters):
    sm_estimator = TensorFlow(entry_point=code_entry,
                              source_dir=code_dir,
                              role=sagemaker_role,
                              instance_type='local',
                              instance_count=1,
                              model_dir='/opt/ml/model',
                              hyperparameters=hyperparameters,
                              output_path='file://{}/model/'.format(os.getcwd()),
                              framework_version='2.2',
                              py_version='py37',
                              script_mode=True)
    return sm_estimator


def sagemaker_local_training(local_estimator, train_data_local):
    local_estimator.fit({'training':train_data_local})
    return local_estimator


def sagemaker_local_deploy(local_estimator):
    local_predictor = local_estimator.deploy(initial_instance_count=1, instance_type='local')
    return local_predictor


if __name__ == '__main__':
    code_entry = 'tf_script.py'
    code_dir = os.getcwd() + '/tf_code/'
    hyperparameters = {'epochs': 5, 'batch_size': 64, 'learning_rate': 0.01, 'drop_rate': 0.2}
    local_estimator = sagemaker_estimator(sagemaker_role, code_entry, code_dir, hyperparameters)

    local_inputs = 'file://{}/data/'.format(os.getcwd())
    local_estimator = sagemaker_local_training(local_estimator, local_inputs)

    # just in case there are other inference containers running in Local Mode,
    # we'll stop them to avoid conflict before deploying our new model locally.
    os.system('docker container stop $(docker container ls -aq) >/dev/null')
    local_predictor = sagemaker_local_deploy(local_estimator)

    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data(path=os.getcwd() + '/data/mnist.npz')
    prediction_results = local_predictor.predict(x_test[:20])['predictions']
    print('Prediction: ', np.argmax(prediction_results, axis=1))
    print('Ground truth: ', y_test[:20])
    local_predictor.delete_endpoint()




