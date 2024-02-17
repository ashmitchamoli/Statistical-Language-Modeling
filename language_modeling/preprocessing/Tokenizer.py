import re

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from preprocessing.Cleaner import Cleaner
from preprocessing.REGEX import tokenizerRegEx

class Tokenizer:
    def __init__(self):
        self.textCleaner = Cleaner()

    def readText(self, inputText : str, readFromFile : bool = False) -> str:
        '''
        Read text from a file specified by path and return it as a string after cleaning.
        '''
        return self.textCleaner.cleanText(inputText, readFromFile)
    
    def sentenceTokenize(self, cleanedText : str) -> list:
        '''
        Return a list of sentences after splitting about punctuations.
        '''
        sentenceList = re.split(tokenizerRegEx['<SEN>'], cleanedText)

        finalList = []
        for i in range(0, len(sentenceList), 2):
            if (sentenceList[i].strip() == ''):
                continue
            finalList.append([])
            finalList[-1].append(sentenceList[i].strip())
            if (i + 1 < len(sentenceList)):    
                finalList[-1].append('\b' + sentenceList[i + 1].strip())
            
        return finalList
    
    def wordTokenize(self, sentenceList : list) -> list:
        '''
        Return a list of sentences after splitting into words.
        '''
        wordList = []
        for sentence in sentenceList:
            wordList.append([])
        
            words = re.split(tokenizerRegEx['<WORD>'], sentence[0])    
            for word in words:
                word = word.strip(' \b')
                if (word != None and word != ''):
                    if (word[0] in ',;' or word in self.textCleaner.openEnclosures):
                        wordList[-1].append('\b' + word)
                    elif (word in self.textCleaner.closeEnclosures):
                        wordList[-1].append(word + '')
                    else:
                        wordList[-1].append(word)
            
            if (len(sentence) > 1):
                wordList[-1].append(sentence[1])

        return wordList

    def tokenize(self, inputText : str, readFromFile : bool = False) -> list[list[str]]:
        '''
        Return a list of lists of tokens after tokenizing the text.
        Set readFromFile True if you want to read the text from a file specified by text.
        '''
        cleanedText = self.readText(inputText, readFromFile)
        sentenceList = self.sentenceTokenize(cleanedText)
        tokenList = self.wordTokenize(sentenceList)

        return tokenList

if __name__ == "__main__":
    tokenizer = Tokenizer()
    # print(tokenizer.tokenize('example.txt', True))
    tokenList = tokenizer.tokenize('./example1.txt', readFromFile=True)
    print(len(tokenList))
    for token in tokenList:
        print(token, "len:", len(token))