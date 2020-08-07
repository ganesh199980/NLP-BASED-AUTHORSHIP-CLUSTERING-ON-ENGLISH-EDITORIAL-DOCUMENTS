import nltk
import re


def POS_Tagger(text):
    pos_tokens = []
    #print(len(text))

    c=0
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    pos = nltk.pos_tag(tokens)
    po = [p[1] for p in pos]
    pos_tokens.append(po)
    return pos_tokens
