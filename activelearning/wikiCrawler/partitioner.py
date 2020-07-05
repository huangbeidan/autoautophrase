import wikipedia
import tools.fileHandler as filehandler
from tqdm import tqdm
import math

from multiprocessing import Lock, Process, Queue, current_process
import time
import queue # imported for using queue.Empty exception



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

def do_job(tasks_to_accomplish, tasks_that_are_done):
    while True:
        try:
            '''
                try to get task from the queue. get_nowait() function will 
                raise queue.Empty exception if the queue is empty. 
                queue(False) function would do the same task also.
            '''
            # task should be like (pid, words)
            task = tasks_to_accomplish.get_nowait()
            pid = task[0]
            words = task[1]
            partition_worker(words, pid)

        except queue.Empty:
            break

        else:
            '''
                if no exception has been raised, add the task completion 
                message to task_that_are_done queue
            '''
            tasks_that_are_done.put("partition: " + str(pid) + ' is done by ' + current_process().name)
            time.sleep(.5)
    return True


def run():
    p_size = 10
    allwords = filehandler.getwords('../../input/wiki_quality.txt', split=False)
    allwords = allwords[:100]
    number_of_task = math.ceil(len(allwords)/p_size)
    number_of_processes = 4
    tasks_to_accomplish = Queue()
    tasks_that_are_done = Queue()
    processes = []


    for i in range(number_of_task):
        # each process responsible for 10000 words
        tasks_to_accomplish.put((i, allwords[i*p_size :  (i+1)*p_size-1]))

    # creating processes
    for w in range(number_of_processes):
        p = Process(target=do_job, args=(tasks_to_accomplish, tasks_that_are_done))
        processes.append(p)
        p.start()

    # completing process
    for p in processes:
        p.join()

    # print the output
    while not tasks_that_are_done.empty():
        print(tasks_that_are_done.get())

    return True


if __name__ == '__main__':
    run()











    # print(sentences==True)
