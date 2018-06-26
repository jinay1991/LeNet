from urllib.request import urlretrieve
import os
import zipfile
import pickle
import numpy as np

def download(url, file):
    """
    Download file from <url>
    :param url: URL to file
    :param file: Local file path
    """
    if not os.path.isfile(file):
        print('Downloading ' + file + '...')
        urlretrieve(url, file)
        print('Download Finished')

def uncompress(file):
    """
    Uncompress features and labels from a zip file
    :param file: The zip file to extract the data from
    """
    zip_ref = zipfile.ZipFile(file, 'r')
    zip_ref.extractall('.')
    zip_ref.close()

def load_data(url):
    download(url, 'traffic-sign-data.zip')
    uncompress('traffic-sign-data.zip')

    training_file = 'train.p'
    validation_file = 'valid.p'
    testing_file = 'test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    X_train = np.concatenate((X_train, X_valid), axis=0)
    y_train = np.concatenate((y_train, y_valid), axis=0)
    np.savetxt("labels.txt", np.unique(y_train))

    print()
    print("Image Shape: {}".format(X_train[0].shape))
    print()
    print("Training Set:   {} samples".format(len(X_train)))
    print("Validation Set: {} samples".format(len(X_valid)))
    print("Test Set:       {} samples".format(len(X_test)))

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="dataset zip url",
                        default='https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip')
    args = parser.parse_args()

    print(args)

    # load data from dataset
    X_train, y_train, X_test, y_test = load_data(args.url)
