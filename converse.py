from language_modeling.models.NGramModel import NGramModel
from time import sleep

N = input("Enter N: ")
smoothingType = input("Enter smoothing type (normal, gt, i): ")
book = "Pride and Prejudice - Jane Austen"
# book = "Ulysses - James Joyce"
model = NGramModel(int(N), f'data/datasets/{book}/train.txt', smoothingType=smoothingType)

# prompt = '"Mr. Bennet, how can you abuse your own children in such a way? You take delight in vexing me.'
prompt = input("Enter prompt: ")
print("\033[1;32m" + prompt, end="\033[0m ")
word = model.generateNextWord(prompt)

while (word != None):
    print(word, end=" ", flush=True)
    prompt += " " + word
    word = model.generateNextWord(prompt)
    sleep(0.1)