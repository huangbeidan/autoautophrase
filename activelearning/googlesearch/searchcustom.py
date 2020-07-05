
if __name__ == '__main__':

    from google import google
    num_page = 1
    search_results = google.search("This is my query", num_page)
    for result in search_results:
        print(result.name)