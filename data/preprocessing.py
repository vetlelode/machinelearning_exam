# preprocesses the data
import re
import random


def process_line(text):
    return re.sub("[\"\n]", "", text)


if __name__ == '__main__':
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
    print("success!")


def split_training_data(data: list, anomalous_data: list, training_size: float, test_bias: float):
    tmp_data, tmp_anomalous_data = list(data), list(anomalous_data)


    training_data, test_data = [], []
    for item in anomalous_data:
        ls = training_data if random.random() < training_size else test_data
        ls.append(item)

    # normal dist:


    return training_data, test_data
