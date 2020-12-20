# preprocesses the data
import re
import random
import numpy as np


REAL_DATA_MAX_N: int = 284315
FAKE_DATA_MAX_N: int = 492


def process_line(text: str) -> str:
    return re.sub("[\"\n]", "", text)


def process_lines(lines: list) -> list:
    return [[float(f) for f in process_line(line).split(",")] for line in lines]


def split_training_data(data: list, anomalous_data: list, f: float = 0.5) -> tuple:
    k = int((1-f) * len(anomalous_data))
    
    test_rdata, training_rdata = sample_split(data, k)
    test_adata, training_adata = sample_split(anomalous_data, k)
    test_data = test_adata + test_rdata
    training_data = training_adata + training_rdata
    random.shuffle(test_data)
    random.shuffle(training_data)
    return (*split_XY(training_data), *split_XY(test_data))


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


def get_dataset(k1: int = 284315, k2: int = 492, f: float = 0.5):
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

    k1 = min(k1, REAL_DATA_MAX_N)
    k2 = min(k2, FAKE_DATA_MAX_N)
    anomalous_data, real_data = [], []
    with open("../data/real.csv", "r") as data_file:
        lines = data_file.readlines()[1:]
        real_data = process_lines(random.sample(lines, k1))

    with open("../data/fake.csv", "r") as data_file:
        lines = data_file.readlines()[1:]
        random.shuffle(lines)
        anomalous_data = process_lines(random.sample(lines, k2))

    return np.array(split_training_data(real_data, anomalous_data, f))


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
