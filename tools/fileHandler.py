def getwords(filename, split = True):
    with open(filename) as f1:
        t5 = f1.read().splitlines()
        if split:
            t5_list = [t.split('\t')[1] for t in t5]
            return t5_list
    return t5


def getsent(filename):
    with open(filename) as f1:
        t5 = f1.read().splitlines()
        return ' '.join(t5)


def writeZipRowsToFile(filename, rows):
    with open(filename, 'w') as fp:
        out = ['\n'.join(f'{i}\t{j}' for i, j in rows)]
        fp.writelines(out)



def getwordswithscore(filename):
    output = {}
    with open(filename) as f1:
        t5 = f1.read().splitlines()
        for t in t5:
            t5_s = t.split('\t')
            output[t5_s[1]] = t5_s[0]
    return output

def writeListToFile(ls, fn, append = False):

    if append:
        with open(fn, 'a') as f:
            for item in ls:
                f.write("%s\n" % item)

    else:
        with open(fn, 'w') as f:
            for item in ls:
                f.write("%s\n" % item)

def integrateFiles():
    listtotal = getwords("../input/section_navyGlossary.txt", split=False) +\
                getwords("../input/section_navyGlossary2.txt", split=False) +\
                getwords("../input/section_phrasepdf.txt", split=False)
      # don't use known phrase for now, we need to keep adding criteria
                # getwords("../outputs/is_known_phrase_nodup.txt", split=False)
    listtotal = list(dict.fromkeys(listtotal))
    print("total quality length: ", len(listtotal))
    writeListToFile(listtotal, "../outputs/allqualityphrase.txt")


if __name__ == '__main__':
    # rows = zip([1,2,3], [4,5,6])
    # writeZipRowsToFile('../tmp/test.txt', rows)

    integrateFiles()