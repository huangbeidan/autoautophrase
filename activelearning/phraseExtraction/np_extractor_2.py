import nltk
from nltk.corpus import stopwords
from tools.fileHandler import getsent, writeListToFile
from activelearning.utils.normalise import Normalizer
from collections import defaultdict


def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        yield subtree.leaves()


def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword. We can increase the length if we want to consider large phrase"""
    accepted = bool(2 <= len(word) <= 40
                    and word.lower() not in stopwords.words('english'))
    return accepted


def get_terms(tree, freq=False):
    nz = Normalizer()
    for leaf in leaves(tree):
        if freq:
            term = [nz.normalise(w) + "%" + t for w, t in leaf if acceptable_word(w)]
        else:
            term = [nz.normalise(w) + "%" + t for w, t in leaf if acceptable_word(w)]

        yield term

def run(freq = False, all_phrase_dict = None):

    text = getsent('/Users/beidan/RASHIP/PDFs-TextExtract/output/section.txt')
    sentence_re = r'(?:(?:[A-Z])(?:.[A-Z])+.?)|(?:\w+(?:-\w+)*)|(?:\$?\d+(?:.\d+)?%?)|(?:...|)(?:[][.,;"\'?():-_`])'
    grammar = r"""
            NBAR:
                {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

            NP:
                {<NBAR>}
                {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
        """
    chunker = nltk.RegexpParser(grammar)
    toks = nltk.regexp_tokenize(text, sentence_re)
    postoks = nltk.tag.pos_tag(toks)
    tree = chunker.parse(postoks)
    terms = get_terms(tree, freq=freq)
    output = []
    for term in terms:
        name_entity = ' '.join(term)
        name_entity = Normalizer().cleanPhrase(name_entity)
        output.append(name_entity)
        if freq:
            all_phrase_dict[name_entity] += 1
    if freq:
        return all_phrase_dict
    return output


if __name__ == '__main__':

    # nltk.download('wordnet')

    output = run()
    print(output)
    print("length of output: " , len(output))
    writeListToFile(output, '../../outputs/np_extract_r3.txt')

    # test = run(freq=True, all_phrase_dict=defaultdict(lambda:0))
    # print(test)

