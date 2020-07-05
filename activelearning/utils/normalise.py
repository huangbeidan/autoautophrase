
import nltk
from nltk.corpus import stopwords
from tools.fileHandler import getsent, writeListToFile
import re

class Normalizer:
    def __init__(self):
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.stemmer = nltk.stem.porter.PorterStemmer()

    def normalise(self,word):
        """Normalises words to lowercase and stems and lemmatizes it."""
        word = word.lower()
        # word = stemmer.stem_word(word) #if we consider stemmer then results comes with stemmed word, but in this case word will not match with comment
        word = self.lemmatizer.lemmatize(word)
        return str(word).strip()

    def acceptable_word(self, word):
        """Checks conditions for acceptable word: length, stopword. We can increase the length if we want to consider large phrase"""
        # accepted = bool(2 <= len(word) <= 40
        #                 and word.lower() not in stopwords)
        accepted = bool(2 <= len(word) <= 40)
        return accepted

    def cleanPhrase(self, phrase):
        # phrase = re.sub(r'(?<=[0-9])\s+(?=(?:st|nd|rd|th)\b)', '', phrase)
        phrase = re.sub(r'(?<=(-\s[a-zA-Z]))\s(?=(?:[a-zA-Z0-9]*)\b)', '', phrase)
        phrase = re.sub(r'^[^a-zA-Z]+', '', phrase)
        for w in phrase.split(' '):
            if len(w) > 1:
                return phrase
        return ''

    def filterPhrases(self, phrases):
        return [t for t in phrases if t != '']



if __name__ == '__main__':
    test0 = '- e lectronic warfare (ew'
    test = ': vi rginia class submarine (vc )/co lumbia class submarine (cl ); d dg'
    print(Normalizer().cleanPhrase(test))

