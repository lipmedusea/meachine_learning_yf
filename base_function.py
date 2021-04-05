def Combinations(L, k):
    """List all combinations: choose k elements from list L"""
    n = len(L)
    result = [] # To Place Combination result
    for i in range(n-k+1):
        if k > 1:
            newL = L[i+1:]
            Comb, _ = Combinations(newL, k - 1)
            for item in Comb:
                item.insert(0, L[i])
                result.append(item)
        else:
            result.append([L[i]])
    return result, len(result)

import itertools
# for i in itertools.product('ABCD', repeat = 2):
#     print(i)
#
# for i in itertools.permutations('ABCD', 2):
#     print(i)
#
# for i in itertools.combinations('ABCD', 2):
#     print(i)
#
# for i in itertools.combinations_with_replacement('ABCD', 2):
#     print(i)
# #不同集合组合
# list1 = [1, 2, 3]
# list2 = [4, 5, 6]
# s = list(itertools.product(list1, list2))

import itertools
import pandas as pd
def combine(param_grid):
    s = []
    for x in param_grid:
        s.append(x)
    lis = []
    for y in s:
        lis.append(param_grid[y])
    para_ori = pd.DataFrame(list(itertools.product(*lis)), columns=s)
    return para_ori


