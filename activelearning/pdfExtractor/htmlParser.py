import re

from bs4 import BeautifulSoup
import lxml.html

## remove some attributes from the html file
def attributesCleaner(filename):

    input = open(filename)
    html_string = ""
    for lines in input.readlines():
        html_string += lines + "\n"

    # Parse the html
    html = lxml.html.fromstring(html_string)

    # Method 2
    for tag in html.xpath('//*[@top]'):
        # For each element with a class attribute, remove that class attribute
        tag.attrib.pop('top')

    for tag in html.xpath('//*[@left]'):
        # For each element with a class attribute, remove that class attribute
        tag.attrib.pop('left')
    for tag in html.xpath('//*[@right]'):
        # For each element with a class attribute, remove that class attribute
        tag.attrib.pop('right')
    for tag in html.xpath('//*[@char]'):
        # For each element with a class attribute, remove that class attribute
        tag.attrib.pop('char')
    for tag in html.xpath('//*[@bottom]'):
        # For each element with a class attribute, remove that class attribute
        tag.attrib.pop('bottom')
    for tag in html.xpath('//figure'):
        # For each element with a class attribute, remove that class attribute
        tag.getparent().remove(tag)

    output_string = lxml.html.tostring(html)

    return output_string

    # file1 = open("file/output_small_filtered.html","w")
    # file1.write(str(output_string))
