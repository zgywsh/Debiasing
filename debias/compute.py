import joblib
import os
import pickle as pkl
import torch


def compute_bias(b):
    bias_ = 0
    # for value in results[b].keys():
    #     print(value)
    for value in results[b].values():

        race = value.max(dim=-1).indices

        unique_elements, counts = torch.unique(race, return_counts=True)
        freq = {
            element.item(): count.item()
            for element, count in zip(unique_elements, counts)
        }
        # print(freq)
        K = len(freq)

        freq_diff_sum = 0
        keys = list(freq.keys())
        if K == 1:
            bias_ += 1
            print("1")
            continue
        for i in range(K):
            for j in range(i + 1, K):
                freq_diff_sum += abs(freq[keys[i]] - freq[keys[j]])

        bias_P = freq_diff_sum / (K * (K - 1) / 2) / len(race)
        print(bias_P)

        bias_ += bias_P
    print(bias_ / len(results[b]))


def compute_Diversity(b):
    D = 0
    for value in results[b].values():

        race = value.max(dim=-1).indices

        unique_elements, counts = torch.unique(race, return_counts=True)
        n = len(unique_elements)
        n = torch.tensor(n)

        total_count = counts.sum().item()
        counts = counts.float() / total_count
        d = 0
        for i in counts.float():
            d += i * torch.log(i)
        kkk = d / (torch.log(1 / n))
        # print(kkk)
        D += kkk

    print(D / len(results[b]))


import numpy as np


def compute_bias_m(b, c):

    bias_ = 0
    for value1, value2 in zip(results[b].values(), results[c].values()):
        gender = value1.max(dim=-1).indices
        race = value2.max(dim=-1).indices

        a_values = gender.unique()
        b_values = race.unique()

        all_combinations = torch.cartesian_prod(a_values, b_values)

        data = torch.stack((gender, race), dim=1)
        counts = torch.zeros(all_combinations.size(0), dtype=torch.int32)

        for i, combo in enumerate(all_combinations):
            counts[i] = (data == combo).all(dim=1).sum()

        # print(all_combinations, counts)

        freq = {
            tuple(element.tolist()): count.item()  # 使用元组作为键
            for element, count in zip(all_combinations, counts)
        }
        K = len(freq)

        freq_diff_sum = 0
        keys = list(freq.keys())
        if K == 1:
            bias_ += 1
            continue

        for i in range(K):
            for j in range(i + 1, K):
                freq_diff_sum += abs(freq[keys[i]] - freq[keys[j]])

        bias_P = freq_diff_sum / (K * (K - 1) / 2) / len(race)
        print(bias_P)

        bias_ += bias_P

    print(bias_ / len(results[b]))


if __name__ == "__main__":
    with open(
        "test_results.pkl",
        "rb",
    ) as f:
        results = pkl.load(f)

    # compute_bias(2)
    # compute_bias(3)
    compute_bias_m(2, 3)

    # compute_Diversity(3)
    # from midu import compute_dis

    # compute_dis(
    #     "/train_outputs/gen_images/images_debias"
    # )
