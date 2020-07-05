from collections import defaultdict

import pdftotree
import time

from htmlParser import attributesCleaner

def parsePDF(input_name, output_name, special_pages):

    #special_pages = [1]


    start1 = time.time()
    data1 = pdftotree.parse(input_name, html_path=None, model_type=None, model_path=None, favor_figures=True,
                            visualize=False, pages = special_pages)
    end1 = time.time()

    # output_name = "file/output_{}.html".format(file_name)
    file1 = open(output_name, "w")
    file1.write(data1)
    time1 = end1 - start1
    print("==== time elapase for parsing small PDF file====")
    print(time1)
    return output_name

def run():

    fn_input = '../../input/RDTEN-books/BA13_mini.pdf'

    # out_inter_path path --- "file/outputpdfs/{}.html"
    outputfile = parsePDF(fn_input, '../../tmp/BAparsed.txt', ['default'])

    finaloutput_string = attributesCleaner(outputfile)

    # out_final_path - "file/outputpdfs/{}_filtered.html"
    file1 = open('../../tmp/BAparsed_fi.txt', "w")
    finaloutput_string_str = str(finaloutput_string)
    finaloutput_string_str = finaloutput_string_str[2: len(finaloutput_string_str) - 1]
    file1.write(finaloutput_string_str)

if __name__ == '__main__':
    run()

