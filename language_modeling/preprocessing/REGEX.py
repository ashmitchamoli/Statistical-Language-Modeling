urlRegExAlt = "(^https?:\\/\\/(?:www\\.)?)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$"
urlRegEx = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
emailRegEx = "[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+"
hashTagRegEx = "(#+[a-zA-Z0-9(_)]{1,})"
mentionRegEx = "(\b@+[a-zA-Z0-9(._)]{1,})"
numbersRegEx = "([0-9]+)"

placeholderRegEx = {
    '<URL>' : urlRegEx,
    '<MAILID>' : emailRegEx,
    '<HASHTAG>' : hashTagRegEx,
    '<MENTION>' : mentionRegEx,
    '<NUM>' : numbersRegEx
}

sentenceRegEx = "([.?!]+)"
wordRegEx = "(<URL>|<MAILID>|<HASHTAG>|<MENTION>|<NUM>|[a-zA-Z0-9]+)"
tokenizerRegEx = {
    '<SEN>' : sentenceRegEx,
    '<WORD>' : wordRegEx
}