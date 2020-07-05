
import tools.fileHandler as filehandler


def run():
    t5_list = filehandler.getwords('../input/t5.txt')
    t6_list = filehandler. getwords('../input/t6.txt')
    # list_diff = [item for item in t5_list if item not in t6_list]
    list_diff = list(set(t5_list) - set(t6_list))
    filehandler.writeListToFile(list_diff, "t4Dt5.txt")

    t5_dict = filehandler.getwordswithscore('../input/t5.txt')
    list_diff_score = [ (t5_dict[t] + '\t' + t) for t in list_diff]

    list_diff_score= sorted(list_diff_score, key = lambda t: t.split('\t')[0], reverse=True)
    print(list_diff_score)

    filehandler.writeListToFile(list_diff_score, "t4Dt5_score.txt")


if __name__ == '__main__':
    run()

