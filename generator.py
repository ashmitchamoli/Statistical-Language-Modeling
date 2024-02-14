from include.models.NGramModel import NGramModel
import sys

# python generator.py <smoothing type> <corpus path> <k>
smoothingType = sys.argv[1]
corpusPath = sys.argv[2]
k = int(sys.argv[3])

text = input("input text: ")

if smoothingType == 'g':
    smoothingType = 'gt'

model = NGramModel(3, corpusPath, smoothingType=smoothingType)

generatedText = model.nextWordPrediction(text)

# print top k words
for word, prob in sorted(generatedText.items(), key=lambda item: item[1], reverse=True)[:k]:
    print(word, prob)