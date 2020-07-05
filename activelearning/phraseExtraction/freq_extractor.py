from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import RegexpParser
from nltk import Tree
from nltk.corpus import stopwords
from collections import defaultdict
from activelearning.phraseExtraction.np_extractor_2 import run
import pickle

import pandas as pd
from tools.fileHandler import getsent, writeListToFile
from activelearning.utils.normalise import Normalizer




def get_continuous_chunks_freq(text, chunk_func=ne_chunk, all_phrases_dict=None):
    if all_phrases_dict is None:
        all_phrases_dict = defaultdict(lambda: 0)

    chunked = chunk_func(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []

    for subtree in chunked:
        if type(subtree) == Tree:
            # if adding the posTag information...
            current_chunk.append(" ".join([Normalizer().normalise(str(token)) + '%' + str(pos) for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            named_entity = Normalizer().cleanPhrase(named_entity)
            all_phrases_dict[named_entity] += 1
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    return all_phrases_dict

if __name__ == '__main__':

    # Defining a grammar & Parser
    NP = "NP: {(<V\w+>|<NN\w?>)+.*<NN\w?>}"
    chunker = RegexpParser(NP)

    # apply to our file
    sent = getsent('/Users/beidan/RASHIP/PDFs-TextExtract/output/section.txt')
    print(sent)

    # way 1 :use custom chunker
    all_phrases_dict = get_continuous_chunks_freq(sent, chunker.parse, None)

    # way 2 : use ner chunker
    all_phrases_dict = get_continuous_chunks_freq(sent, chunk_func=ne_chunk, all_phrases_dict=all_phrases_dict)

    # way 3: use trees
    all_phrases_dict = run(freq=True, all_phrase_dict=all_phrases_dict)

    # remove the empty keys
    all_phrases_dict = {k: v for k, v in all_phrases_dict.items() if v is not None}

    print(all_phrases_dict)
    list1 = list(all_phrases_dict.keys())
    list2 = list (all_phrases_dict.values())
    all_phrases_freq_list = ['\t'.join(map(str, i)) for i in zip(list2, list1)]

    # sort the list
    all_phrases_freq_list = sorted(all_phrases_freq_list, key=lambda x: int(x.split('\t')[0]), reverse=True)
    all_phrases_freq_list2 = sorted(all_phrases_freq_list, key=lambda x: (x.split('\t')[1]), reverse=False)

    # write output to file
    writeListToFile(all_phrases_freq_list, '../../outputs/np_extract_with_freq.txt')
    writeListToFile(all_phrases_freq_list2, '../../outputs/np_extract_with_freq_alpha.txt')

    # also pickle the file for future use
    pickle.dump(dict(all_phrases_dict), open('../../tmp/phrases_freq', 'wb'))