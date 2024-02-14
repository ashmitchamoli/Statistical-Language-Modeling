from include.models.NGramModel import NGramModel
import sys

# commandline arguments: python language_model.py <smoothing type> <corpus path>
smoothingType = sys.argv[1]
corpusPath = sys.argv[2]

if smoothingType == 'g':
    smoothingType = 'gt'
model = NGramModel(3, corpusPath, smoothingType=smoothingType)

sentence = input("input sentence: ")
print("score: ", model.sentenceProbability(sentence))