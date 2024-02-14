from include.preprocessing.Tokenizer import Tokenizer

text = input("your text: ")

tokenizer = Tokenizer()
tokenizedText = tokenizer.tokenize(inputText=text)

print(tokenizedText)