import tools.fileHandler as fileHandler
import math
from collections import defaultdict
from activelearning.utils.normalise import Normalizer
from activelearning.learning.learning import removePosFromPhrase
import re
from google import google
import activelearning.phraseExtraction.knownphrase as knownphrase
import json
import csv
from activelearning.featureExtraction.hierclusteringscipy import getLabels


def getPhrases():
    return fileHandler.getwordswithscore('../../outputs/np_extract_with_freq_alpha.txt')


def ngrams(phrase, n):
    phrase = phrase.split(' ')
    output = []
    for i in range(len(phrase) - n + 1):
        output.append(phrase[i:i + n])
    return output


class Phrase:

    def __init__(self, phrase, freq):
        self.phrase = phrase
        self.freq = freq
        self.subgram_freq = []
        self.max_subgram_freq = 0
        self.supports = []
        self.min_support = math.inf
        self.mutual_confidence = -1
        self.partial_order = 0
        self.f_len = 0
        self.f_avg_wordlength = 0
        self.f_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.f_completeness = 0
        self.f_postagUniqueness = 0
        self.label = None
        self.is_know_phrase = 0

    def print_it(self):
        info = ("phrase: {} \n".format(self.phrase),
                "freq: {} \n".format(self.freq),
                "subgram_freq: {} \n".format(self.subgram_freq),
                "max_subgram_freq: {} \n".format(self.max_subgram_freq),
                "partial_order: {} \n".format(self.partial_order),
                "supports: {} \n".format(self.supports),
                "min_support: {} \n".format(self.min_support),
                "mutual_confidence: {} \n".format(self.mutual_confidence),
                "f_len: {} \n".format(self.f_len),
                "f_avg_wordlength: {} \n".format(self.f_avg_wordlength),
                "f_pos: {} \n".format(self.f_pos),
                "f_completeness: {} \n".format(self.f_completeness),
                "f_postagUniqueness: {} \n".format(self.f_postagUniqueness),
                "is_known_phrase: {} \n".format(self.is_know_phrase))
        print(info)

    def write_it(self):
        output = {}
        output['phrase'] = self.phrase
        output['phrase_freq'] = self.freq
        output['phrase_length'] = self.f_len
        output['subgram_freq'] = self.subgram_freq
        output['max_subgram_freq'] = self.max_subgram_freq
        output['partial_order'] = self.partial_order
        output['supports'] = self.supports
        output['min_support'] = self.min_support
        output['mutual_confidence'] = self.mutual_confidence
        output['f_avg_wordlength'] = self.f_avg_wordlength
        output['f_pos'] = self.f_pos
        output['f_completeness'] = self.f_completeness
        output['f_postagUniqueness'] = self.f_postagUniqueness
        output['is_known_phrase'] = self.is_know_phrase
        return output

    def unfold_it(self):
        # print("pattern: ", self.phrase)
        min_support = self.min_support
        if self.min_support == math.inf:
            min_support = -1
        is_know_phrase = self.is_know_phrase
        if math.isnan(int(self.is_know_phrase)):
            is_know_phrase = 0
        output = [self.phrase.replace(",", "`sep`"), int(self.freq), self.f_len, self.max_subgram_freq, self.partial_order,
                  min_support, self.mutual_confidence, self.f_avg_wordlength, *self.f_pos, self.f_completeness,
                  self.f_postagUniqueness, int(is_know_phrase)]
        return output

        # return ["freq", "f_len", "max_subgram_freq", "partial_order", "min_support", "mutual_confidence",
        #         "f_avg_wordlength",
        #         "f_pos_nn", "f_pos_prop", "f_pos_verb", "f_pos_adj", "f_pos_deter", "f_pos_empty", "f_completeness",
        #         "f_postagUniqueness", "is_know_phrase"]





class FeatureMining:

    def __init__(self):
        # initially all the phrases have POS tags
        self.all_phrases_raw_pos = getPhrases()
        # list of phrases with postags
        self.all_phrases_list_pos = list(self.all_phrases_raw_pos.keys())

        # all_phrases_raw is the dictionary without postags
        # 注意这里的长度为5481 < 5833 删去了DUPLICATE POS TAGS的情况，所以KNOWN PHRASE顺序答应要用另一个ARRAY
        self.all_phrases_raw, self.scoresdict = self.removePosTagFromDict(self.all_phrases_raw_pos)
        self.all_phrases_list = list(self.all_phrases_raw.keys())

        # patterns without postags either
        self.patterns = [Phrase(k, self.all_phrases_raw[k]) for k in self.all_phrases_raw.keys()]
        self.unique_tokens_map = self.tokens2id()
        self.phrase2patternid_map = self.phrase2patternid()


    def extractFeaturesWithPattern(self):
        for pattern in self.patterns:
            t = pattern.phrase
            pattern.f_len = len(t.split(' '))
            pattern.f_avg_wordlength = self.calAvgWordLength(t)
            # pattern.f_pos = self.countPos(t)
            pattern.f_incompleteness = 1 if self.areParanthesisBalanced(t) else 0
            pattern.f_countUpper = self.countCapital(t)
        #self.checkPosTagsUniqueness()
        self.miningMaxSubGramFreq()
        self.miningEpisodeScore(4)
        self.miningOrdality()
        self.miningMutualConfidence()
        self.checkingKnownPhrases()
        self.miningPosTagFreq()

    def phrase2patternid(self):
        output = {}
        for i in range(len(self.patterns)):
            phrase = self.patterns[i].phrase
            if phrase not in output:
                output[phrase] = []
            output[phrase].append(i)
        return output


    def modifyPatternPostag(self, phrase, posDistribution, posUnique=True):
        if phrase not in self.phrase2patternid_map:
            return #only 1 key miss, that's fine
        patternids = self.phrase2patternid_map[phrase]
        for patternid in patternids:
            self.patterns[patternid].f_pos = posDistribution
            if not posUnique:
                 self.patterns[patternid].f_postagUniqueness = 1


    def miningPosTagFreq(self):
        seen = set()
        for phrase_pos in self.all_phrases_list_pos:
            posUnique = True
            phrase = self.removePosFromPhrase(phrase_pos)
            if phrase not in seen:
                seen.add(phrase)
            else:
                posUnique = False

            tmp = self.countPos(phrase_pos)
            self.modifyPatternPostag(phrase, tmp, posUnique = posUnique)


    def miningMaxSubGramFreq(self):
        for pattern in self.patterns:
            n_gram = len(pattern.phrase.split(' ')) - 1
            for i in range(n_gram + 1):
                if i < 2: continue
                ith_grams = ngrams(pattern.phrase, i)
                max_cnt = 0
                for gram in ith_grams:
                    pr = ' '.join(gram)
                    if pr in self.all_phrases_list:
                        max_cnt = max(max_cnt, int(self.all_phrases_raw[pr]))
                pattern.subgram_freq.append(max_cnt)
            if len(pattern.subgram_freq) > 0:
                pattern.max_subgram_freq = max(pattern.subgram_freq)

    def miningEpisodeScore(self, gap_level):
        for pattern in self.patterns:
            words = pattern.phrase.split(' ')
            i = len(words) - gap_level - 1
            while i < len(words):
                subgram = words[:i]
                score = math.inf
                subphrase = ' '.join(subgram)
                if subphrase in self.all_phrases_list and int(self.all_phrases_raw[subphrase]) != 0:
                    # print("miningEpisodeScore hit!")
                    score = int(self.all_phrases_raw[pattern.phrase]) / int(self.all_phrases_raw[subphrase])
                pattern.supports.append(round(score, 2))
                i += 1
            pattern.min_support = min(pattern.supports)

    def miningMutualConfidence(self):
        for pattern in self.patterns:
            words = pattern.phrase.split(' ')
            i = 2
            j = 2
            # i j are the length not index, both starting from 1
            support_l = 0
            support_r = 0
            support = int(self.all_phrases_raw[pattern.phrase])
            while (i + j) <= len(words):
                # inner loop for j
                subphrase_left = ' '.join(words[:i])
                if subphrase_left in self.all_phrases_list and int(self.all_phrases_raw[subphrase_left]) != 0:
                    support_l = max(support_l, int(self.all_phrases_raw[subphrase_left]))

                while (i + j) <= len(words):
                    subgram_right = words[-j:]
                    subphrase_right = ' '.join(subgram_right)
                    if subphrase_right in self.all_phrases_list and int(self.all_phrases_raw[subphrase_right]) != 0:
                        support_r = max(support_r, int(self.all_phrases_raw[subphrase_right]))
                    j += 1
                i += 1
            if support_r != 0 and support_l != 0:
                pattern.mutual_confidence = round((support_l / support_r + support / support_r) / 2,3)

    def miningOrdality(self):
        # 如果打乱顺序后在其他地方出现过，则为partial order
        for token in self.unique_tokens_map:
            if len(self.unique_tokens_map[token]) > 1:
                indexes = self.unique_tokens_map[token]
                for idx in indexes:
                    pattern = self.patterns[idx]
                    pattern.partial_order = 1
                    # pattern.print_it()

    def removePosTagFromDict(self, all_phrases_raw_pos):
        all_phrases_raw = {}
        scoresraw = fileHandler.getwords('../../outputs/knownphrase/knowphrase_all_v2.txt', split=False)
        scoresdict = {}
        print("all_phrases_raw_pos: ", all_phrases_raw_pos)
        i = 0
        for phrase_pos_key in all_phrases_raw_pos:
            phrase_key = re.sub(r'%[A-Z]+\b', '', phrase_pos_key)
            all_phrases_raw[phrase_key] = all_phrases_raw_pos[phrase_pos_key]
            if phrase_key not in scoresdict:
                scoresdict[phrase_key] = scoresraw[i]
            else:
                tmp = scoresdict[phrase_key]
                scoresdict[phrase_key] = max(tmp, scoresraw[i])
            i += 1

        return all_phrases_raw, scoresdict

    def checkPosTagsUniqueness(self):
        seen = set()
        for pattern in self.patterns:
            phrase = pattern.phrase
            pp = removePosFromPhrase(phrase)
            if pp in seen:
                pattern.f_postagUniqueness = 1
            else:
                seen.add(pp)

    def checkzeroscores(self, scores):
        run_dict = {}
        score0 = 0
        score1 = 0
        score2 = 0
        score3 = 0
        score4 = 0
        for i in range(len(scores)):
            if scores[i] == '0':
                run_dict[i] = 0
                score0 += 1
            elif scores[i] == '1':
                score1 += 1
            elif scores[i] == '2':
                score2 += 1
            elif scores[i] == '3':
                score3 += 1
            elif scores[i] == '4':
                score4 += 1
        print("score distribution ==== \n score0: {}, score1: {}, "
              "score2: {}, score3: {}, score4: {}".format(score0, score1, score2, score3, score4))
        return run_dict

    def checkingKnownPhrases(self, repeat=False):

        rawscores = fileHandler.getwords('../../outputs/knownphrase/knowphrase_all_v2.txt', split=False)
        run_dict = self.checkzeroscores(rawscores)
        cnt = 0
        # this step is to refill the 0 values due to google block
        if repeat:
            while (len(run_dict) > round(0.000 * len(rawscores))) and (cnt < 10):
                rawscores = knownphrase.secondrun(run_dict, rawscores)
                run_dict = self.checkzeroscores(rawscores)
                cnt += 1
            fileHandler.writeListToFile(rawscores, '../../outputs/knownphrase/knowphrase_all.txt')

        # update all the patterns with rawscore
        # ppp = []
        # for i in range(len(self.patterns)):
        #     self.patterns[i].is_know_phrase = rawscores[i]
        #     ppp.append(self.patterns[i].phrase)
        #     fileHandler.writeListToFile(ppp, '../../tmp/phrase_check_fm.txt')

        # made the score into dic
        ## attention: migrate the step into removePosFromDict()
        # print("len of scores dict: ", len(self.scoresdict))
        # print("len of patterns: ", len(self.patterns))

        assert len(self.scoresdict) == len(self.patterns)
        for i in range(len(self.patterns)):
            phrase = self.patterns[i].phrase
            self.patterns[i].is_know_phrase = self.scoresdict[phrase]

        return rawscores

    ## ================================================

    ### Helper Functions #####

    ## ================================================
    def tokens2id(self):
        output = {}
        for i in range(len(self.all_phrases_list)):
            token = ' '.join(sorted(self.all_phrases_list[i].split(' ')))
            if token == '': continue
            if token not in output:
                output[token] = []
            output[token].append(i)
        # print(output)
        return output

    def addToSet(self, out_set, phrase):
        for p in phrase.split(' '):
            if "%" in p:
                out_set.add(str(p).split('%')[1])
        return out_set

    def getAllposTags(self, phrases):
        output = set()
        for phrase in phrases:
            self.addToSet(output, phrase)
        print(output)

    def countPos(self, phrase):
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
        words_len = len(phrase.split(' '))
        output = [noun, prop, verb, adj, deter, empty, other]
        return [round(t / words_len, 3) for t in output]

    def areParanthesisBalanced(self, phrase):
        return ("(" in phrase and ")" not in phrase) or ("(" not in phrase and ")" in phrase)

    def removePosFromWord(self, word):
        word = re.sub('%.*', '', word)
        return word

    def removePosFromPhrase(self, phrase):
        words = phrase.split(' ')
        words = [self.removePosFromWord(t) for t in words]
        return ' '.join(words)

    def calAvgWordLength(self, phrase):
        words = phrase.split(' ')
        words = [self.removePosFromWord(t) for t in words]
        return round(sum(len(word) for word in words) / len(words), 2)

    def countCapital(self, phrase):
        word = phrase.split('%')[0]
        if not word: return 0
        return len(re.findall(r'[A-Z]', word))

    def writeFeaturesToFile(self):
        with open("../../outputs/features_results_all.txt", 'w') as fn:
            for pattern in self.patterns:
                info_dict = pattern.write_it()
                fn.write(json.dumps(info_dict))
                fn.write('\n')

    def _getFeaturesName(self):
        return ["phrase", "freq", "f_len", "max_subgram_freq", "partial_order", "min_support", "mutual_confidence", "f_avg_wordlength",
                "f_pos_nn", "f_pos_prop","f_pos_verb","f_pos_adj","f_pos_deter","f_pos_empty","f_pos_other", "f_completeness",
                "f_postagUniqueness", "is_know_phrase"]

    def writeFeaturesToCSV(self):
        with open("../../outputs/features_results_all_withphrase.csv", 'w') as fn:
            fn_writer = csv.writer(fn, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            fn_writer.writerow(self._getFeaturesName())
            for pattern in self.patterns:
                tmp = pattern.unfold_it()
                fn_writer.writerow(tmp)


    ### For feature mining and sample clustering ######
    def clustering(self):
        label_dict = getLabels(method='ward')

        group1 = [self.patterns[i].phrase for i in label_dict if label_dict[i] == 0]
        group2 = [self.patterns[i].phrase for i in label_dict if label_dict[i] == 1]
        group3 = [self.patterns[i].phrase for i in label_dict if label_dict[i] == 2]
        group4 = [self.patterns[i].phrase for i in label_dict if label_dict[i] == 3]
        group5 = [self.patterns[i].phrase for i in label_dict if label_dict[i] == 4]
        # group6 = [self.patterns[i].phrase for i in label_dict if label_dict[i] == 5]
        # group7 = [self.patterns[i].phrase for i in label_dict if label_dict[i] == 6]
        print("clustering=====")
        print("group1 length: " , len(group1))
        print("group2 length: " , len(group2))
        print("group3 length: " , len(group3))
        print("group4 length: " , len(group4))
        print("group5 length: " , len(group5))
        # print("group6 length: " , len(group6))
        # print("group7 length: " , len(group7))

        fileHandler.writeListToFile(group1, "../../outputs/features_group1_part.txt")
        fileHandler.writeListToFile(group2, "../../outputs/features_group2_part.txt")
        fileHandler.writeListToFile(group3, "../../outputs/features_group3_part.txt")
        fileHandler.writeListToFile(group4, "../../outputs/features_group4_part.txt")
        fileHandler.writeListToFile(group5, "../../outputs/features_group5_part.txt")
        # fileHandler.writeListToFile(group6, "../../outputs/features_group6_complete.txt")
        # fileHandler.writeListToFile(group7, "../../outputs/features_group7_complete.txt")



if __name__ == '__main__':
    fm = FeatureMining()
    # fm.miningMaxSubGramFreq(fm.all_phrases)
    # print(fm.all_phrases_raw)
    fm.extractFeaturesWithPattern()

    ## ====== print out known phrases ==========
    # kp4 = []
    # cnt = 0
    # for pattern in fm.patterns:
    #     if int(pattern.is_know_phrase) == 4:
    #         kp4.append(pattern.phrase)
    #     cnt += 1
    # print("len in fm: ", cnt)
    # fileHandler.writeListToFile(kp4, '../../tmp/kp4.txt')

    # fm.patterns[125].print_it()
    #fm.writeFeaturesToFile()

    fm.writeFeaturesToCSV()
    cnt = 0
    # for pattern in fm.patterns:
    #     if(pattern.f_postagUniqueness == 1):
    #         cnt += 1
    # print("postag uniquess? ", cnt)

    #fm.clustering()




