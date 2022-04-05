import os

def search_names(path, extension='.csv'):
    """
    Search for files with a given extension in a given path.
    """
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(extension):
                matches.append(os.path.join(root, filename))
    return matches
