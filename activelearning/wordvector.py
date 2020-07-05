import spacy



# python3 -m spacy download en_core_web_md

if __name__ == '__main__':

    nlp = spacy.load("en_core_web_md")
    tokens = nlp("dog cat banana afskfsd")

    for token1 in tokens:
        for token2 in tokens:
            if token1 != token2:
                print(token1.text, token2.text, token1.similarity(token2))