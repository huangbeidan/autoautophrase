from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import RegexpParser
from nltk import Tree
from nltk.corpus import stopwords

import pandas as pd
from tools.fileHandler import getsent, writeListToFile
from activelearning.utils.normalise import Normalizer


def get_continuous_chunks(text, chunk_func=ne_chunk):
    chunked = chunk_func(pos_tag(word_tokenize(text)))
    #print(chunked)
    continuous_chunk = []
    current_chunk = []

    for subtree in chunked:
        if type(subtree) == Tree:
            # if adding the posTag information...
            current_chunk.append(" ".join([nz.normalise(str(token)) + '%' + str(pos) for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            named_entity = Normalizer().cleanPhrase(named_entity)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    return continuous_chunk


if __name__ == '__main__':
    # Defining a grammar & Parser
    NP = "NP: {(<V\w+>|<NN\w?>)+.*<NN\w?>}"
    chunker = RegexpParser(NP)

    nz = Normalizer()

    # apply to our file
    sent = getsent('/Users/beidan/RASHIP/PDFs-TextExtract/output/section.txt')
    print(sent)
    # way 1 : use ner chunker
    chunks = get_continuous_chunks(sent)
    # way 2 :use custom chunker
    #chunks = get_continuous_chunks(sent, chunker.parse)
    print(chunks)
    writeListToFile(chunks, '../../outputs/np_extract_r1.txt')


