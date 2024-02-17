import numpy as np

class LinearInterpolationModel:
    def __init__(self, allNGramFreqs: dict[int, dict[tuple[str], dict[str, int]]]) -> None:
        """
        allNGramFreqs[1] = unigramFreqs
        allNGramFreqs[2] = bigramFreqs
        ... 
        allNGramFreqs[N] = N-gramFreqs
        """
        self.__N = len(allNGramFreqs) # for N-gram model we need freqs from unigram to (N-1)-gram
        self.lambdas = self.__computeLambdas(allNGramFreqs)

    def __computeLambdas(self, allNGramFreqs: list[dict[tuple[str], dict[str, int]]]) -> list[float]:
        """
        Computes the lambdas for the linear interpolation model.
        .. math::
            `P(w_n | w_{n-1}...w_1) = \Lambda_1 P_{ML}(w_n) + \Lambda_2 P_{ML}(w_n | w_{n-1}) + ... + \Lambda_N P_{ML}(w_n | w_{n-1}...w_1)`
        """
        lambdas = np.zeros(self.__N+1)
        lambdas[0] = -1 # dummy value
        nGramFreqs = allNGramFreqs[self.__N]

        for n_1Gram in nGramFreqs:
            for word in nGramFreqs[n_1Gram]:
                if word == '<__total__>':
                    continue
                # compute the lambdas
                nGram = n_1Gram + (word, )
                probs = np.zeros(shape=self.__N+1)
                probs[0] = -1 # dummy value
                for i in range(1, self.__N+1):
                    iGramFreq = allNGramFreqs[i]
                    iGram = nGram[-i:]
                    context = iGram[:-1]
                    word_ = iGram[-1]
                    if iGramFreq[context].get('<__total__>', 0) <= 1 or word_ not in iGramFreq[context]:
                        probs[i] = 0
                    else:
                        probs[i] = (iGramFreq[context][word_] - 1) / (iGramFreq[context]['<__total__>'] - 1)

                lambdas[np.argmax(probs)] += nGramFreqs[nGram[:-1]][nGram[-1]]

        return lambdas[1:] / np.sum(lambdas[1:])