import tools.fileHandler as fileHandler
import re
import string
import wikipedia
from google import google
from activelearning.utils.normalise import Normalizer
import math


def getAllPhrases():
    output = set(fileHandler.getwords('../../outputs/np_extract_r1.txt', split=False)).union(
        set(fileHandler.getwords('../../outputs/np_extract_r2.txt', split=False))).union(
        set(fileHandler.getwords('../../outputs/np_extract_r3.txt', split=False))
    )
    return list(output)

def addToSet(out_set, phrase):
    for p in phrase.split(' '):
        if "%" in p:
            out_set.add(str(p).split('%')[1])
    return out_set


def getAllposTags(phrases):
    output = set()
    for phrase in phrases:
        addToSet(output, phrase)
    print(output)


def countPos(phrase):
    noun = 0
    prop = 0
    verb = 0
    adj = 0
    deter = 0
    empty = 0
    other = 0

    for word in phrase.split(' '):
        if '%' not in str(word):
            empty += 1
            continue
        posTag = str(word).split('%')[1]
        if posTag == '':
            empty += 1
        elif posTag in ['NNP', 'NNPS', 'NNS', 'NN']:
            noun += 1
        elif posTag in ['IN']:
            prop += 1
        elif posTag in ['VBG', 'VBD', 'VBN', 'VBZ', 'VBP', 'VB']:
            verb += 1
        elif posTag in ['JJ']:
            adj += 1
        elif posTag in ['DT']:
            deter += 1
        else:
            other += 1
    return [noun, prop, verb, adj, deter, empty, other]


def areParanthesisBalanced(phrase):
    return ("(" in phrase and ")" not in phrase) or ("(" not in phrase and ")" in phrase)


def calAvgWordLength(phrase):
    words = phrase.split(' ')
    words = [removePosFromWord(t) for t in words]
    return round(sum(len(word) for word in words) / len(words), 2)

def checkPosTagsUniqueness(phrases):
    seen = set()
    output = []
    for phrase in phrases:
        pp = removePosFromPhrase(phrase)
        if pp in seen:
            output.append(1)
        else:
            output.append(0)
            seen.add(pp)
    return output


def extractFeatures(phrases):
    f_len = [len(t.split(' ')) for t in phrases]
    f_avg_wordlength = [calAvgWordLength(t) for t in phrases]
    f_pos = [countPos(t) for t in phrases]
    f_incompleteness = [1 if areParanthesisBalanced(t) else 0 for t in phrases]
    f_countUpper = [countCapital(t) for t in phrases]
    f_postagUniqueness = checkPosTagsUniqueness(phrases)

    rows = zip(f_len, f_pos, f_incompleteness, f_countUpper)
    # fileHandler.writeZipRowsToFile('../../outputs/feature_result1.txt', rows)
    print(list(rows)[0])

    print(phrases)
    print("f_len: ", f_len)
    print("f_avg_wordlength: ", f_avg_wordlength)
    print("f_pos: ", f_pos)
    print("f_completeness: ", f_incompleteness)
    print("f_countUpper: ", f_countUpper)
    print("f_postagUniqueness: ", f_postagUniqueness)


def removePosFromWord(word):
    word = re.sub('%.*', '', word)
    return word

def removePosFromPhrase(phrase):
    words = phrase.split(' ')
    words = [removePosFromWord(t) for t in words]
    return ' '.join(words)

from difflib import SequenceMatcher


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def checkgoogle(phrase):
    phrase = [removePosFromWord(t) for t in phrase.split(' ')]
    # flag = wikipedia.search('missile defense network power-projection platform')
    query = ' '.join(phrase)
    search_results = google.searchC(query, 1)
    search_results = [Normalizer().normalise(t) for t in search_results]
    print(search_results)
    if len(search_results) == 0: return 0

    ss = 0
    for res in search_results:
        #print("res: ", res)
        if str(query) in str(res):
            ss = max(ss, 4)
        score = similar(query, str(res))
        if score > 0.8:
            ss = max(ss, 3)
        elif score > 0.3:
            ss = max(ss, 2)
        else:
            ss = max(ss, 1)

    return ss


def countCapital(phrase):
    word = phrase.split('%')[0]
    if not word: return 0
    return len(re.findall(r'[A-Z]', word))


def run():
    # only write once, in order to keep data consistency
    # output = getAllPhrases()
    # output = [t for t in output if t != '']
    # output = sorted(output)
    # fileHandler.writeListToFile(output, '../../outputs/np_extract_all_normalized.txt')

    ## read from file to make the result consistent
    output = fileHandler.getwords('../../outputs/np_extract_all_normalized.txt', split=False)

    extractFeatures(output)
    # getAllposTags(output)

    # print(output)
    # print("length: ", len(output))


if __name__ == "__main__":
    run()
