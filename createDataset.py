import os
import token
import tokenize
from sklearn.model_selection import train_test_split

from language_modeling.preprocessing.Tokenizer import Tokenizer

dataPath = "data/books"

txt_files = [file for file in os.listdir(dataPath) if file.endswith(".txt")]

tokenizer = Tokenizer()
for file in txt_files:
    cleanedText = tokenizer.readText(inputText=os.path.join(dataPath, file), readFromFile=True)
    sentences = tokenizer.sentenceTokenize(cleanedText)
    
    # randomly split into train and test without shuffling
    train, test = train_test_split(sentences, test_size=1000/len(sentences), random_state=42, shuffle=False)
    
    # save train and test
    with open(f"data/datasets/{file[:-4]}/train.txt", "w") as f:
        for sentence in train:
            f.write(" ".join(sentence) + "\n")

    with open(f"data/datasets/{file[:-4]}/test.txt", "w") as f:
        for sentence in test:
            f.write(" ".join(sentence) + "\n")