import base64
import sys

with open(sys.argv[1], "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
    with open('base64.txt', 'wb+') as fp:
        fp.write(encoded_string)