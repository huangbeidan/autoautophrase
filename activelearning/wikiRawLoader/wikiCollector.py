from wikipedia import Wikipedia
from Wiki2Plain import Wiki2Plain


if __name__ == '__main__':

    lang = 'simple'
    wiki = Wikipedia(lang)

    try:
        raw = wiki.article('Uruguay')
        print(raw)
    except:
        raw = None

    if raw:
        wiki2plain = Wiki2Plain(bytes(raw).decode("utf-8"))
        content = wiki2plain.text
        print(content)