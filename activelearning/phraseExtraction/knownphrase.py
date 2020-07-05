import tools.fileHandler as filehandler
from activelearning.utils.normalise import Normalizer
from google import google
from difflib import SequenceMatcher
import re


def getPhrases():
    return filehandler.getwordswithscore('../../outputs/np_extract_with_freq_alpha.txt')


def simplerun():
    phrases = list(getPhrases().keys())
    st1 = 4633
    end1 = 5633
    tmp = phrases[st1: end1]
    # print("tmp: ", tmp)
    output = [checkgoogle(t) for t in tmp]
    filehandler.writeListToFile(output, "../../outputs/knownphrase/knowphrase_{}.txt".format(5), append=False)


def secondrun(run_dict, scores):
    # run_dict {idx, score}
    phrases = list(getPhrases().keys())
    print("length of run dict: ", len(run_dict))
    for idx in run_dict:
        res = checkgoogle(phrases[idx])
        if int(res) != 0:
            scores[idx] = res
    return scores

def anotherrun(repeat = False):
    # run_dict {idx, score}
    rawscores = filehandler.getwords('../../outputs/knownphrase/knowphrase_all_v2.txt', split=False)
    run_dict = checkzeroscores(rawscores)
    cnt = 0
    # this step is to refill the 0 values due to google block
    if repeat:
        while (len(run_dict) > round(0.0 * len(rawscores))) and (cnt < 10):
            rawscores = secondrun(run_dict, rawscores)
            run_dict = checkzeroscores(rawscores)
            cnt += 1
        filehandler.writeListToFile(rawscores, '../../outputs/knownphrase/knowphrase_all_v2.txt')

    # update all the patterns with rawscore
    return rawscores

def checkzeroscores(scores):
        run_dict = {}
        for i in range(len(scores)):
            if scores[i] == '0':
                run_dict[i] = 0
        return run_dict

def integratelist():

    scores = filehandler.getwords('../../outputs/knownphrase/knowphrase_0.txt', split=False) + \
             filehandler.getwords('../../outputs/knownphrase/knowphrase_1.txt', split=False) + \
             filehandler.getwords('../../outputs/knownphrase/knowphrase_2.txt', split=False) + \
             filehandler.getwords('../../outputs/knownphrase/knowphrase_3.txt', split=False) + \
             filehandler.getwords('../../outputs/knownphrase/knowphrase_4.txt', split=False) + \
             filehandler.getwords('../../outputs/knownphrase/knowphrase_5.txt', split=False) + \
             filehandler.getwords('../../outputs/knownphrase/knowphrase_6.txt', split=False) + \
             filehandler.getwords('../../outputs/knownphrase/knowphrase_7.txt', split=False) + \
             filehandler.getwords('../../outputs/knownphrase/knowphrase_8.txt', split=False) + \
             filehandler.getwords('../../outputs/knownphrase/knowphrase_9.txt', split=False) + \
             filehandler.getwords('../../outputs/knownphrase/knowphrase_10.txt', split=False) + \
             filehandler.getwords('../../outputs/knownphrase/knowphrase_11.txt', split=False) + \
             filehandler.getwords('../../outputs/knownphrase/knowphrase_12.txt', split=False)
    print(scores)
    print("len of scores: ", len(scores))
    # pick out those zero score items and run again
    filehandler.writeListToFile(scores, '../../outputs/knownphrase/knowphrase_all_v2.txt')


def run(n):
    phrases = list(getPhrases().keys())

    ed1 = 5150
    for i in range(n):
        try:
            st1 = ed1
            ed1 = st1 + 700
            tmp = phrases[st1: ed1]
            # print("tmp: ", tmp)
            output = [checkgoogle(t) for t in tmp]
            filehandler.writeListToFile(output, "../../outputs/knownphrase/knowphrase_{}.txt".format(i+12),
                                        append=False)
        except Exception as e:
            print(e)
            continue


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def removePosFromWord(word):
    word = re.sub('%.*', '', word)
    return word


def checkgoogle(phrase, test=False):
    # flag = wikipedia.search('missile defense network power-projection platform')
    phrase = [removePosFromWord(t) for t in phrase.split(' ')]
    query = ' '.join(phrase)
    search_results = google.searchC(query, 1)
    search_results = [Normalizer().normalise(t) for t in search_results]
    print(search_results)
    if len(search_results) == 0: return 0

    ss = 0
    for res in search_results:
        # print("res: ", res)
        if str(query) in str(res):
            ss = max(ss, 4)
        score = similar(query, str(res))
        if score > 0.8:
            ss = max(ss, 3)
        elif score > 0.3:
            ss = max(ss, 2)
        else:
            ss = max(ss, 1)
    if test:
        print("score: ", ss)

    return ss

def printHighQPhrases(debug = False):
    phrases = list(getPhrases().keys())
    phrases = [' '.join([removePosFromWord(t) for t in phrase.split(' ')]) for phrase in phrases]
    scores = filehandler.getwords('../../outputs/knownphrase/knowphrase_all_v2.txt', split=False)
    output = []
    for i in range(len(scores)):
        if int(scores[i]) == 4:
            output.append(phrases[i])
    print("len of high quality phrase: ", len(output))
    if debug:
        print(phrases)
        print("length of total phrases: ", len(phrases))
    filehandler.writeListToFile(output, '../../tmp/kp4.txt')

def writePhrasesWithoutDuplicates():
    phrases = filehandler.getwords("../../tmp/kp4.txt", split=False)
    phrases = list(dict.fromkeys(phrases))
    phrases = [t for t in phrases if len(t.split(' '))>1]
    filehandler.writeListToFile(phrases, "../../outputs/is_known_phrase_nodup.txt")

if __name__ == '__main__':
    # simplerun()
    # integratelist()
    # phrase = "warfighting system information security concern project effort research activity"
    # checkgoogle(phrase, test=True)
    # printHighQPhrases()
    # integratelist()

    #anotherrun(repeat=True)

    printHighQPhrases(debug=True)
    writePhrasesWithoutDuplicates()

   # run(1)



