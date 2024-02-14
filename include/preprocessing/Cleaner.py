import re
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from preprocessing.REGEX import placeholderRegEx

class Cleaner:
    '''
    The function of the Cleaner class is to ready the text for tokenization.
    The provided functions can be used to detect URLs, Hashtags, Mentions, Numbers, Mail IDs and replace them with appropriate placeholders.
        Mail IDs: <MAILID>
        URLs: <URL>
        Hashtags: <HASHTAG>
        Mentions: <MENTION>
        Numbers: <NUM>
    This is needed to be done before tokenization because the tokenization process splits the text into tokens based on characters such as spaces, commas, periods, etc.
    If the text is not cleaned before tokenization, the tokens will be split at the special characters in the URLs, Hashtags, Mentions, Numbers, Mail IDs, etc. and the task will become needlessly complex.
    '''
    def __init__(self) -> None:
        self.enclosures = [ ('(', ')'), 
                            ('[', ']'), 
                            ('{', '}')]
        self.openEnclosures = [ bracket[0] for bracket in self.enclosures ]
        self.closeEnclosures = [ bracket[1] for bracket in self.enclosures ]

        self.replacementPrecedence = [
            '<MENTION>',
            '<MAILID>',
            '<URL>',
            '<HASHTAG>',
            '<NUM>'
        ]
        
    def readText(self, path : str) -> str:
        '''
        Read text from a file specified by path and return it as a string.
        '''
        with open(path, 'r') as f:
            return f.read()
    
    def writeText(self, path : str, text : str) -> None:
        '''
        Write text to a file specified by path.
        '''
        with open(path, 'w') as f:
            f.write(text)
    
    def replacePattern(self, token : str, regEx : str, replacement : str) -> str:
        '''
        Replaces all occurences of the pattern specified by regEx in the token with the replacement string.
        '''
        return re.sub(regEx, replacement, token)
    
    def cleanText(self, inputText : str, readFromFile : bool = False) -> str:
        '''
        Return the text after replacing URLs, Mail IDs, Hashtags, Mentions, Numbers with appropriate placeholders.
        Set readFromFile True if you want to read the text from a file specified by text.
        '''
        if (readFromFile):
            text = self.readText(inputText)
        else:
            text = inputText

        # replace newlines with spaces
        tokens = text.split()

        cleanedText = ''

        for token in tokens:
            for placeholder in self.replacementPrecedence:
                token = self.replacePattern(token, placeholderRegEx[placeholder], placeholder)
            cleanedText += token + ' '

        return cleanedText
    
if __name__ == "__main__":
    cleaner = Cleaner()
    text = cleaner.cleanText("./example.txt", readFromFile=True)
    print(text)