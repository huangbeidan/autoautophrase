import wikipedia
import tools.fileHandler as filehandler
from tqdm import tqdm

def try_word(word):
    try:
        sent = wikipedia.summary(word, sentences=2, auto_suggest=False)
        print("sucessfully found the term of word: ", word)
    except Exception as ex:
        print(ex)

def partition_worker(words, pid):
    work = []
    with open ('../../tmp/wiki_quality_sentences_{}.txt'.format(pid), 'w') as f:
        for word in tqdm(words):
            try:
                sent = wikipedia.summary(word, sentences=2, auto_suggest=True)
                sent = sent.replace('\n', ' ')
                f.write("%s\n" % sent)
                work.append(word)
            except Exception as ex:
                print(ex)

    filehandler.writeListToFile(work, "../../outputs/wiki_work_{}.txt".format(pid))



if __name__ == '__main__':
    #print (wikipedia.summary("new york city", sentences=2, auto_suggest=False))

    words = filehandler.getwords('../../input/wiki_quality.txt', split=False)
    # print(words)

    partition_worker(words[3401:3500], 1 )


    # try_word("Henry Billings Brown")








    # print(sentences==True)
