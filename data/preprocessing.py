# preprocesses the data
import re


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
    with open("../data/fraud_data.csv", "w") as data_file:
        data_file.write(labels + "\n" + ("\n".join(fraud_data)))

    with open("../data/real_data.csv", "w") as data_file:
        data_file.write(labels + "\n" + ("\n".join(real_data)))
    print("success!")
