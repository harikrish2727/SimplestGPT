import requests
import os
import pathlib
# url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


# with open("input.txt", "w", encoding="utf-8") as f:
#     f.write(text)



# with open("/content/input.txt","r",encoding="utf-8") as f:
#   input_text = f.read()

def get_data(url):
    if not pathlib.Path("input.txt").exists():
        data = requests.get(url).text
        with open("input.txt", "w", encoding="utf-8") as f:
            f.write(data)
    return

def read_data():
    if not pathlib.Path("input.txt").exists():
        print("input.txt does not exist. Downloading...")
        try:
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            get_data(url)
        except FileNotFoundError:
            raise FileNotFoundError("input.txt does not exist.")
    
    print("input.txt already exists. Reading from file...")
    with open("input.txt","r",encoding="utf-8") as f:
        data = f.read()
    return data



