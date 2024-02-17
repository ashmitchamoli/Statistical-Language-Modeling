from scipy.stats import linregress
from typing import Callable
import numpy as np

class GoodTuringModel:
    def __init__(self, nGramFreqs : dict[tuple[str], dict[str, int]]) -> None:
        self.__gtModel = self.__generateGTModel(nGramFreqs)
        
    def rStar(self, r : int) -> float:
        return self.__gtModel(r)

    def __generateGTModel(self, nGramFreqs : dict[tuple[str], dict[str, int]]) -> Callable[[int], float]:
        """
        returns a lambda function F such that F(r) = r*
        """
        N = {}
        for nGram in nGramFreqs:
            for word in nGramFreqs[nGram]:
                if word == '<__total__>':
                    continue
                r = nGramFreqs[nGram][word]
                N[r] = N.get(r, 0) + 1

        # sort the dictionary by keys
        N = dict(sorted(N.items(), key=lambda x : x[0]))

        # print(N)
        if len(N) < 2:
            raise Exception("Not enough data to generate Good-Turing model.")

        # calculate Z_r's
        Z = {}
        nonZeroNr = list(N.keys())
        for i in range(len(nonZeroNr)):
            r = nonZeroNr[i]
            if r == 1:
                Z[1] = N[1] / (0.5 * nonZeroNr[i+1])
            elif i == len(nonZeroNr) - 1:
                Z[r] = N[r] / (r - nonZeroNr[i-1])
            else:
                Z[r] = N[r] / (0.5 * (nonZeroNr[i+1] - nonZeroNr[i-1]))

        # fit a linear regression model to log(Z_r) vs log(r)
        result = linregress(x = np.log(list(Z.keys())),
                            y = np.log(list(Z.values())))
        b = result.slope
        # print(b, result.intercept)
        rStar = lambda r : (r+1) * np.exp(b * (np.log((r+1)/r))) if r != 0 else N[1]

        return rStar
        