import os
from collections import Counter


def find_param(file1_path):
    cishu = 0

    file2_path = "/gradient_info.txt"

    with open(file1_path, "r") as f1:
        indices = {int(line.strip()) for line in f1 if line.strip().isdigit()}

    param_dict = {}
    with open(file2_path, "r") as f2:
        for line in f2:
            if ":" in line:
                key, value = line.split(":")
                start, end = map(int, value.split(" - "))
                param_dict[key] = (start, end)
    # print(len(param_dict))

    param_names = []
    for index in indices:
        for key, (start, end) in param_dict.items():
            if index > start and index <= end:
                param_names.append(key)

    # print(len(param_names))

    param_counts = Counter(param_names)
    param_train = []

    for param_name, count in param_counts.items():
        if count > cishu:
            param_train.append(param_name)
    return param_train


if __name__ == "__main__":
    param_train = find_param(500)
    print(len(param_train))
