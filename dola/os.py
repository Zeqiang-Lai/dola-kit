import os
import re

def filename(path):
    return os.path.splitext(os.path.basename(path))[0]

def fileext(path):
    return os.path.splitext(path)[1]


def make_valid_filename(input_string):
    # Replace spaces with underscores
    filename = input_string.replace(' ', '_')

    # Remove characters not allowed in filenames
    filename = re.sub(r'[\\/:*?"<>|]', '', filename)

    # Limit the filename length (adjust this as needed)
    max_length = 255
    if len(filename) > max_length:
        filename = filename[:max_length]

    return filename
