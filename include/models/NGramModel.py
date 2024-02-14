from itertools import chain
from multiprocessing import context
from time import sleep
import pickle
import hashlib
from urllib.request import proxy_bypass
import numpy as np

import sys
import os

from torch import le

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from preprocessing.Tokenizer import Tokenizer
from models.GoodTuringModel import GoodTuringModel
from models.LinearInterpolationModel import LinearInterpolationModel

EPSILON = 1e-5
EPSILON_I = 1e-5
EPSILON_G = 1e-7

class NGramModel:
    def __init__(self, N : int, corpusPath : str, readFromFile : bool = True, smoothingType : str = 'normal') -> None:
        """
        smoothingType: Options: 'normal', 'gt', 'interpolation'. Default: 'normal'.
        """
        self.__N = N
        self.__corpusPath = corpusPath
        
        self.tokenizer = Tokenizer()
        tokenizedText = self.tokenizer.tokenize(inputText=corpusPath, readFromFile=readFromFile)
        self.__tokenList = list(chain.from_iterable(tokenizedText))
        
        self.__vocabulary = set(self.__tokenList)
        self.__corpusSize = len(self.__tokenList)
        self.__smoothingType = smoothingType
        self.__nGramFreqs = self.__generateNgramsFreqs(N)
        self.__allNGramFreqs = self.__generateAllNGramFreqs(N)
        self.__tokenFreqs = self.__allNGramFreqs[1][()]
        # print(self.__corpusSize)
        # print(self.__nGramFreqs[('and', 'not')]['offensive'], self.__nGramFreqs[('not', 'offensive')]['to'])

        self.__gtModel = None
        self.__gtModels = {}
        self.__interpolationModel = None
        
        if self.__smoothingType == 'gt':
            for i in range(1, self.__N+1):
                self.__gtModels[i] = GoodTuringModel(self.__allNGramFreqs[i])
            self.__gtModel = self.__gtModels[self.__N]


        if self.__smoothingType == 'i':
            self.__interpolationModel = LinearInterpolationModel(self.__allNGramFreqs)
            # print(self.__interpolationModel.lambdas)

        self.__totalNGrams = self.__corpusSize - self.__N + 1
        
    def __generateAllNGramFreqs(self, N : int) -> dict[int, dict[tuple[str], dict[str, int]]]:
        """
        allNGramFreqs[1] = unigramFreqs
        allNGramFreqs[2] = bigramFreqs
        ...
        allNGramFreqs[N] = N-gramFreqs
        """
        allNGramFreqs = {}
        for i in range(1, N+1):
            allNGramFreqs[i] = self.__generateNgramsFreqs(i)
        return allNGramFreqs

    def __generateNgramsFreqs(self, N) -> dict[tuple[str], dict[str, int]]:
        """
        Returns a dictionary D where D[nGram][word] = frequency of seeing word  after nGram
        Additionally, D[nGram]['<__total__>'] = total occurances of nGram in the corpus.
        """
        # check if frequency file for this N exists
        filePath = f"./cache/{hashlib.sha256(self.__corpusPath.encode()).hexdigest()}_freqs_{N}.pkl"
        if os.path.isdir('./cache') == False:
            os.mkdir('./cache')

        if os.path.isfile(filePath):
            with open(filePath, 'rb') as f:
                return pickle.load(f)
        else:
            f = open(filePath, 'x')
            f.close()

        nGramFreqs = {}
        for i in range(self.__corpusSize - N + 1):
            nGram = tuple(self.__tokenList[i : i + N - 1])
            word = self.__tokenList[i + N - 1]

            if nGram not in nGramFreqs:
                nGramFreqs[nGram] = {}
                nGramFreqs[nGram]['<__total__>'] = 0

            if word not in nGramFreqs[nGram]:
                nGramFreqs[nGram][word] = 0
            
            nGramFreqs[nGram][word] += 1
            nGramFreqs[nGram]['<__total__>'] += 1

        with open(filePath, 'wb') as f:
            pickle.dump(nGramFreqs, f)

        return nGramFreqs

    def nextWordPrediction(self, text : str) -> dict[str, float]:
        """
        Given an input text, generates the next word using the N-gram model.
        """
        tokenizedInput = self.tokenizer.tokenize(inputText=text, readFromFile=False)
        tokenizedInput = list(chain.from_iterable(tokenizedInput))

        # need atleast N-1 tokens to generate next word
        if (len(tokenizedInput) < self.__N - 1):
            return None
        
        context = tuple(tokenizedInput[-self.__N + 1 : ])

        if self.__smoothingType == 'normal':
            return self.__normalNGramPrediction(context)

        if self.__smoothingType == 'gt':
            return self.__gtPrediction(context)
        
        if self.__smoothingType == 'i':
            return self.__interpolationPrediction(tuple(['<PAD>'] * (self.__N - 1 - len(context))) + context)

    def generateNextWord(self, text : str) -> str:
        return self.__returnBestWord(self.nextWordPrediction(text))

    def __returnBestWord(self, probDist : dict[str, float]) -> str:
        """
        Returns the word with the highest probability in the probability distribution.
        """
        if probDist is None:
            return None
        
        probs = np.array(list(probDist.values()))
        words = list(probDist.keys())
        probs = probs / np.sum(probs)

        # randomly sample word with probability given by probDist
        return np.random.choice(words, p=probs)

        # return word with highest probability
        # return words[np.argmax(probs)]

    def __normalNGramPrediction(self, nGram : tuple[str]) -> str:
       # have never seen this nGram before
        if nGram not in self.__nGramFreqs:
            return None

        candidateWords = self.__nGramFreqs[nGram]
        
        probDist = {}
        for word in self.__vocabulary:
            if word not in candidateWords:
                probDist[word] = 1 / (candidateWords['<__total__>'] + len(self.__vocabulary))
            else:
                probDist[word] = (candidateWords[word] + 1) / (candidateWords['<__total__>'] + len(self.__vocabulary))

        # pick the word with the highest probability
        # return self.__returnBestWord(probDist)
        return probDist

    def __gtPrediction(self, context : tuple[str]) -> str:
        # have never seen this nGram before
        if context not in self.__nGramFreqs:
            probDist = {}
            for word in self.__vocabulary:
                probDist[word] = (self.__tokenFreqs.get(word, 0) + 1) / (self.__tokenFreqs['<__total__>'] + len(self.__vocabulary))
            return probDist

        candidateWords = self.__nGramFreqs[context]
        probDist = {}
        for word in self.__vocabulary:
            probDist[word] = self.__gtModel.rStar(candidateWords.get(word, 0))
            if word not in candidateWords:
                probDist[word] /= (len(self.__vocabulary) - len(candidateWords) + 1)
        
        probs = np.array(list(probDist.values()))
        probs = probs / np.sum(probs)

        probDist = dict(zip(self.__vocabulary, probs))
        # pick the word with the highest probability
        return probDist

    def __MLEProbDist(self, context : tuple[str], N : int) -> dict[str, float]:
        if N != 1:
            context = context[-N+1:]
        if N == 1:
            context = ()

        nGramFreqs = self.__allNGramFreqs[N]

        candidateWords = nGramFreqs.get(context, {})

        probs = np.zeros(len(self.__vocabulary)+1)
        i = 0
        for word in self.__vocabulary:
            if word not in candidateWords:
                probs[i] = 0
            else:
                probs[i] = candidateWords[word] / candidateWords['<__total__>']
            i += 1

        probs[-1] = EPSILON # for <EOG> token
        probs = probs / np.sum(probs)
        return probs

    def __interpolationPrediction(self, context : tuple[str]) -> str:
        probDist = np.zeros(shape=(self.__N, len(self.__vocabulary)+1))
        
        for i in range(1, self.__N+1):
            probDist[i-1, :] = self.__MLEProbDist(context, i)
        
        probDist = np.sum(probDist * self.__interpolationModel.lambdas.reshape(-1, 1), axis=0)

        # pick the word with the highest probability
        return dict(zip(list(self.__vocabulary) + ['<EOG>'], probDist[:-1]))

    def __MLEProb(self, context : tuple[str], word : str, N : int) -> float:
        """
        Returns a 2 sized array where prob[0] is the probability of the word and prob[1] is the probability of the <EOG> token.
        """
        nGram = (context + (word,))[-N:]
        context = nGram[:-1]

        nGramFreqs = self.__allNGramFreqs[N]
        
        candidateWords = nGramFreqs.get(context, {})

        prob = np.zeros(2)
        if word not in candidateWords:
            prob[0] = 0
        else:
            prob[0] = candidateWords[word] / candidateWords['<__total__>']
        prob[-1] = EPSILON

        return prob 

    def __computeProb(self, context : tuple[str], word : str) -> float:
        """
        Compute the probability of the word given the context.
        """
        if word not in self.__vocabulary:
            if self.__smoothingType == 'gt':
                return EPSILON_G
            if self.__smoothingType == 'i':
                return EPSILON_I
            else: 
                return EPSILON

        finalProb = 0
        if self.__smoothingType == 'i':
            if len(context) < self.__N-1:
                context = tuple(['<PAD>'] * (self.__N - len(context) - 1)) + context

            prob = np.zeros(shape=(self.__N, 2))
            
            for i in range(1, self.__N+1):
                prob[i-1, :] = self.__MLEProb(context, word, i)
            
            prob = np.sum(prob * self.__interpolationModel.lambdas.reshape(-1, 1), axis=0)

            finalProb = prob[0]

        if self.__smoothingType == 'gt':
            contextSize = len(context)
            nGramFreqs = self.__allNGramFreqs[contextSize + 1]
            gtModel = self.__gtModels[contextSize + 1]

            if context not in nGramFreqs:
                return self.__tokenFreqs[word] / self.__corpusSize
            
            candidateWords = nGramFreqs[context]
            denom = 0
            for candidateWord in candidateWords:
                if candidateWord == '<__total__>':
                    continue
                denom += gtModel.rStar(candidateWords[candidateWord])
            
            totalZeroFreqWords = len(self.__vocabulary) - len(candidateWords) + 1
            finalProb = gtModel.rStar(candidateWords.get(word, 0)) / (denom + gtModel.rStar(0))
            if word not in candidateWords:
                finalProb /= totalZeroFreqWords

        if self.__smoothingType == 'normal':
            if context not in self.__nGramFreqs:
                return self.__tokenFreqs[word] / self.__corpusSize
            
            candidateWords = self.__nGramFreqs[context]
            finalProb = candidateWords.get(word, 1) / (candidateWords['<__total__>'] + len(self.__vocabulary))

        return finalProb
    
    def sentenceProbability(self, sentence : str) -> float:
        sentence = tuple(self.tokenizer.tokenize(inputText=sentence)[0])

        logProb = 0
        for i in range(len(sentence)):
            prob = 0
            if sentence[i] not in self.__vocabulary:
                # print("Unknown word: " + sentence[i])
                logProb += np.log(EPSILON)
                continue
            
            prob = self.__computeProb(sentence[max(0, i - self.__N + 1) : i], sentence[i])
            # print(sentence[i], prob)
            logProb += np.log(prob)

        return np.exp(logProb)

    def sentencePerplexity(self, sentence : list[str]) -> float:
        """
        Return the perplexity score of the sentence.
        """
        logPerp = 0
        for i in range(len(sentence)):
            nGram = tuple(sentence[max(0, i - self.__N + 1) : i+1])
            context = nGram[:-1]
            word = nGram[-1]

            prob = self.__computeProb(context, word)
            logPerp += np.log(prob)

        return np.exp(-logPerp / len(sentence))

    def computePerplexity(self, testCorpusPath : str) -> float:
        text = self.tokenizer.readText(testCorpusPath, readFromFile=True)
        sentences = self.tokenizer.tokenize(inputText=text)
        # testVocab = set(list(chain.from_iterable(sentences)))
        # print(testVocab.difference(self.__vocabulary))
        
        f = open(f'PnP{self.__smoothingType}.txt', 'w')
        
        perplexity = 0
        numSentences = 0
        for sentence in sentences:
            sentencePerplexity = self.sentencePerplexity(sentence)
            if (sentencePerplexity != None):
                perplexity += sentencePerplexity
                f.write(' '.join(sentence) + ' ' + str(sentencePerplexity) + '\n')
                numSentences += 1

        print(f"Sentences Picked: {numSentences} / {len(sentences)} = {numSentences / len(sentences) * 100}%")
        return perplexity / numSentences

if __name__ == '__main__':
    N = input("Enter N: ")
    smoothingType = input("Enter smoothing type (normal, gt, i): ")
    # book = "Pride and Prejudice - Jane Austen"
    book = "Ulysses - James Joyce"
    
    model = NGramModel(int(N), f'../../data/datasets/{book}/train.txt', smoothingType=smoothingType)

    # prompt = '"Mr. Bennet, how can you abuse your own children in such a way? You take delight in vexing me.'
    # prompt = input("Enter prompt: ")
    # print("\033[1;32m" + prompt, end="\033[0m ")
    # word = model.generateNextWord(prompt)

    # while (word != None):
    #     print(word, end=" ", flush=True)
    #     prompt += " " + word
    #     word = model.generateNextWord(prompt)
    #     sleep(0.1)

    # testPerp = model.computePerplexity(f'../../data/datasets/{book}/test.txt')
    trainPerp = model.computePerplexity(f'../../data/datasets/{book}/train.txt')

    # print("Test Perplexity: ", testPerp)
    print("Train Perplexity: ", trainPerp)
