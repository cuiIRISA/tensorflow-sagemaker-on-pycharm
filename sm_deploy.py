from sagemaker.tensorflow import TensorFlowModel
from tensorflow.keras import datasets
import numpy as np
import os
import boto3
import json
from time import gmtime, strftime


#https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/deploying_tensorflow_serving.html#deploying-directly-from-model-artifacts

sagemaker_role = 'arn:aws:iam::707684582322:role/service-role/AmazonSageMaker-ExecutionRole-20191024T163188'


def sagemaker_deploy(model_artifacts_s3, sagemaker_role, initial_instance_count, instance_type, endpoint_name):
    tf_model = TensorFlowModel(model_data=model_artifacts_s3, role=sagemaker_role, framework_version='2.2')
    predictor = tf_model.deploy(initial_instance_count=initial_instance_count,
                                instance_type=instance_type,
                                endpoint_name=endpoint_name,
                                wait=True)
    return predictor


def test_predictor(predictor):
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data(path=os.getcwd() + '/data/mnist.npz')
    x_test = x_test / 255.0

    prediction_results = predictor.predict(x_test[:20])['predictions']
    print('Prediction: ', np.argmax(prediction_results, axis=1))
    print('Ground truth: ', y_test[:20])

def test_endpoint(endpoint_name):
    client = boto3.client('sagemaker-runtime')
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data(path=os.getcwd() + '/data/mnist.npz')
    x_test = x_test / 255.0

    for index in range(len(x_test[:20])):
        payload = json.dumps(x_test[index].tolist())
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            Body=payload,
            ContentType='application/json'
        )
        print('Ground truth label: ', y_test[index])
        print('Prediction Endpoint: ', response['Body'].read())


if __name__ == '__main__':
    model_artifacts_s3 ='s3://sagemaker-eu-west-1-707684582322/tf-mnist-training-16-14-43-05/output/model.tar.gz'
    initial_instance_count = 1
    instance_type = 'ml.c5.xlarge'
    endpoint_name = 'tensorflow-inference-{}'.format(strftime("%d-%H-%M-%S", gmtime()))
    print(endpoint_name)
    # the deployment of endpoint might take few minutes. The test prediction will run after endpoint is up
    predictor = sagemaker_deploy(model_artifacts_s3, sagemaker_role, initial_instance_count, instance_type, endpoint_name)

    # test predictor with SageMaker SDK
    test_predictor(predictor)

    # invoke endpoint with Boto3 SDK
    test_endpoint(endpoint_name)

    # don't forget ot delete the endpoint
    predictor.delete_endpoint()