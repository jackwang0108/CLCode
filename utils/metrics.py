import numpy as np


def backward_transfer(results: list[float]) -> float:
    n_tasks = len(results)
    li = list()
    for i in range(n_tasks - 1):
        li.append(results[-1][i] - results[i][i])

    return np.mean(li)


def forward_transfer(results: list[float], random_results: list[float]) -> float:
    n_tasks = len(results)
    li = list()
    for i in range(1, n_tasks):
        li.append(results[i-1][i] - random_results[i])

    return np.mean(li)


def forgetting(results: list[float]) -> float:
    n_tasks = len(results)
    li = list()
    for i in range(n_tasks - 1):
        results[i] += [0.0] * (n_tasks - len(results[i]))
    np_res = np.array(results)
    max_res = np.max(np_res, axis=0)
    for i in range(n_tasks - 1):
        li.append(max_res[i] - results[-1][i])

    return np.mean(li)
