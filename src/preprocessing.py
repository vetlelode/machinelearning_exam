# preprocesses the data
import re
import random
import numpy as np


# Prevalence of fraud: 492 : 284315

REAL_DATA_MAX_N: int = 284315
FAKE_DATA_MAX_N: int = 492


def process_line(text: str) -> str:
    return re.sub("[\"\n]", "", text)


def process_lines(lines: list) -> list:
    return np.asarray([[float(f) for f in process_line(line).split(",")] for line in lines])


def split_training_data(
        inliers   : list, 
        outliers  : list, 
        f: float  = 0.5, 
        train_size = 0.8
        ) -> tuple:
    
    inlier_split_index = int((1-train_size)*len(inliers))
    outlier_split_index = int((1-f)*len(outliers))
    
    random.shuffle(inliers)
    
    test_inliers, train_inliers   = sample_split(inliers, inlier_split_index)
    test_outliers, train_outliers = sample_split(outliers, outlier_split_index)
    
    train_set = train_inliers + train_outliers
    test_set = test_inliers + test_outliers
    
    random.shuffle(test_set)
    random.shuffle(train_set)
    return (*split_XY(train_set), *split_XY(test_set))


def split_XY(data: list) -> tuple:
    X, Y = [], []
    for item in data:
        X += [item[:-1]]
        Y += [item[-1]]
    return X, Y


def sample_split(data: list, k: int) -> tuple:
    tmp = list(data)
    random.shuffle(tmp)
    n = min(len(data),k)
    if n == 0: 
        return [], tmp
    return tmp[:n], tmp[n:]


def get_dataset(
        sample: int = 284315,
        pollution: float = 0.5, 
        train_size:float = 0.8
        ):
    """
    Fetches the dataset and splits into training data and test data
    :param k1: the amount of entries to read from the real data
    :param k2: the amount of entries to read from the anomalous data
    :param f: how much of the anomalous data should be in the training data (0<=f<=1)
    :return: training_data (X,Y), test_data (X, Y)
    """
    if files_absent():
        print("preprocessing files. This may take a while")
        preprocess_files()
        print("done!")

    sample = min(sample, REAL_DATA_MAX_N)
    anomalous_data, real_data = [], []
    with open("../data/real.csv", "r") as data_file:
        lines = data_file.readlines()[1:]
        real_data = process_lines(random.sample(lines, sample))
        real_data = clean_data(real_data)

    with open("../data/fake.csv", "r") as data_file:
        lines = data_file.readlines()[1:]
        random.shuffle(lines)
        anomalous_data = process_lines(lines)
        anomalous_data = clean_data(anomalous_data)
    return np.array(split_training_data(real_data, anomalous_data, pollution, train_size))

def clean_data(data):
    # Second to last column is of exponential order
    # Cleaning this up by taking the logarithm improves the results slightly
    data[:,-2]=np.log(np.add(data[:,-2],1e-8))
    return data

def preprocess_files():
    with open("../data/creditcard.csv", "r") as data_file:
        lines = data_file.readlines()
        labels = process_line(lines[0])
        fraud_data, real_data = [], []
        for line in lines[1:]:
            processed_line = process_line(line)
            if processed_line[-1] == "1":
                fraud_data.append(processed_line)
            else:
                real_data.append(processed_line)
    with open("../data/fake.csv", "w") as data_file:
        data_file.write(labels + "\n" + ("\n".join(fraud_data)))

    with open("../data/real.csv", "w") as data_file:
        data_file.write(labels + "\n" + ("\n".join(real_data)))


def files_absent():
    try:
        with open("../data/real.csv", "r"):
            pass
        with open("../data/fake.csv", "r"):
            pass
    except IOError:
        return True
    else:
        return False


if __name__ == '__main__':
    if input("preprocess the files? y/n ") == "y":
        preprocess_files()
        print("success!")
    else:
        print("ok")
        print(len(get_dataset(100, 100)))
