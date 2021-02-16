import tensorflow as tf
import os
import argparse
import numpy as np
import matplotlib

# When the training job finishes, the container will be deleted including its file system with exception of the
# /opt/ml/model and /opt/ml/output folders.
# Use /opt/ml/model to save the model checkpoints.
# These checkpoints will be uploaded to the default S3 bucket.

# SageMaker default SM_MODEL_DIR=/opt/ml/model
if os.getenv("SM_MODEL_DIR") is None:
    os.environ["SM_MODEL_DIR"] = os.getcwd() + '/model'

# SageMaker default SM_OUTPUT_DATA_DIR=/opt/ml/output
if os.getenv("SM_OUTPUT_DATA_DIR") is None:
    os.environ["SM_OUTPUT_DATA_DIR"] = os.getcwd() + '/output'

# SageMaker default SM_CHANNEL_TRAINING=/opt/ml/input/data/training
if os.getenv("SM_CHANNEL_TRAINING") is None:
    os.environ["SM_CHANNEL_TRAINING"] = os.getcwd() + '/data'



def mnist_training(args):
    print('Training dataset is stored here :', os.listdir(path=args.train))
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path=args.train + '/mnist.npz')
    x_train, x_test = x_train / 255.0, x_test / 255.0

    dense_hidden = args.dense_hidden
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(dense_hidden, activation='relu'),
        tf.keras.layers.Dropout(args.drop_rate),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    loss_fn = tf.keras.losses.sparse_categorical_crossentropy
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, validation_data=(x_test, y_test))
    return model


def test_trained_mode(model_path,output_path):
    model = tf.keras.models.load_model(model_path)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path=args.train + '/mnist.npz')
    x_test = x_test / 255.0
    prediction_results = model.predict(x_test)

    print('Prediction: ', np.argmax(prediction_results, axis=1))
    print('Ground truth: ', y_test)
    results = np.stack((y_test, np.argmax(prediction_results, axis=1)), axis=1)
    output_path = output_path + '/prediction.csv'
    print(output_path)
    np.savetxt(output_path, results, delimiter=',')


# SageMaker default SM_MODEL_DIR=/opt/ml/model
if os.getenv("SM_MODEL_DIR") is None:
    os.environ["SM_MODEL_DIR"] = os.getcwd() + '/model'

# SageMaker default SM_OUTPUT_DATA_DIR=/opt/ml/output
if os.getenv("SM_OUTPUT_DATA_DIR") is None:
    os.environ["SM_OUTPUT_DATA_DIR"] = os.getcwd() + '/output'

# SageMaker default SM_CHANNEL_TRAINING=/opt/ml/input/data/training
if os.getenv("SM_CHANNEL_TRAINING") is None:
    os.environ["SM_CHANNEL_TRAINING"] = os.getcwd() + '/data'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))

    # hyperparameters are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--drop_rate', type=float, default=0.2)
    parser.add_argument('--dense_hidden', type=int, default=128)



    args, _ = parser.parse_known_args()
    print(args)

    # training the neural net
    model = mnist_training(args)

    # save the model
    # it seems that it's important to have a numerical name for your folder:
    model_path = args.model_dir + '/1'
    print('The model will be saved at :', model_path)
    model.save(model_path)
    print('model saved')

    # test the model and output the prediction
    test_trained_mode(model_path, args.output_dir)


