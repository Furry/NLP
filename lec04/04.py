import re
from bs4 import BeautifulSoup
import requests
import nltk

sent = "This is a test sentence that contains 11 numbers & \
it was created on 09/02/2020 evening. \
Can you extract some useful information? this (10−31−2222) \
is not a date though! Nothing new about it :) \
1990−34−12 is another way. Try to extract email \
ids like fhamid@ncf.edu, pioneer111@yahoo.com, \
or binary@gmail.com."

# No clue why i did this but oh well
exps = [
    (r'[Tt]his', "this"),
    (r"\w", "chars"),
    (r"\w+", "words"),
    (r"\W+", "special_chars"),
    (r"\d", "digits"),
    (r"\d+", "numbers"),
    (r"\d+/\d+/\d+", "dates"),
    (r"\d{2}[/−]\d{2}[/−]\d{4}|\d{4}[/−]\d{2}[/−]\d{2}", "various_dates"),
    (r"[nN]\w+", "words_with_an_n"),
    (r"\s[nN]\w+", "words_with_space_and_n"),
    (r"\b[cC]\w+", "boundary_words"),
    (r"\w+[-]?\w+", "hwords"),
    (r"\w+[\s|-|/]\w+", "bigrams"),
    (r"\w+\@\w+.\w{3}", "emails"),
]

########
# 1.14 #
########
string = "This test001 is a simple test. \
You should be able to parse it by 4:50 PM. \
If needed, check the references. The regular \
5 time formats may also look like 12:12:12. \
12xyz or 12.34 is not a valid variable name in \
C/C++/Java. Python is a way more flexible \
than Java. Python may accept 12xyz as a valid name. \
Earlier, people used to consider \
C as the state−of−the art for learning a programming \
language. Now, C is mostly used for low−level \
programming. Extracting urls like \
(https://docs.python.org/3/library/re.html) is not easy. \
We are using Python 3.5 for our class."

# 1.) Time Information
print("Time: ", re.findall(r"\d+:\d+(?::\d+)?", string))

# 2.) Digits along with alphanumeric characters
print("Word + Digits: ", re.findall(r"\w+?[\d]+\w+", string))

# 3.) Words that start with l or L
print("l or L: ", re.findall(r"\b[lL]\w+", string))

# 4.) Words of at least length 4 that start with l or L
print("l or L 4 long: ", re.findall(r"\b[lL]\w{4,}", string))

# 5.) Words that contain at least two vowels
print("Two vowels: ", re.findall(r"\w+[aeiou]{2}\w+", string))

# 6.) Words that start with an uppercase letter
print("Start With Upper: ", re.findall(r"[A-Z]\w+", string))

# 7.) Numbers (decimals and floats)
print("Numbers (decimal/float): ", re.findall(r"\d[\d.]*", string))

# 8.) Extract URLs
print("Urls: ", re.findall(r"(?:https?:\/\/)?\w+\.\w+\.\w+[\w\/.]+", string))

# 9.) Extract domain names
# The URL regex matches the entire URL, so I just reused the regex with a capture group around the domain :D
print("Domain Names: ", re.findall(r"(?:https?:\/\/)?\w+\.(\w+)\.\w+", string))

# 10.) Extract domain from email address
print("Domain in Email", re.findall(r"\w+@(\w+)\.\w+", string))

# 2
r = requests.get("https://en.wikipedia.org/wiki/Virus").text
parsed = BeautifulSoup(r)
content = parsed.find(id="bodyContent").text

# Get libs
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Run the POS tagger on the text
tokens = nltk.word_tokenize(content)
# Remove all the non-words
tokens = [token for token in tokens if re.match(r"\w+", token)]
# Only include words 4 or more characters long
tokens = [token for token in tokens if len(token) >= 4]
tagged = nltk.pos_tag(tokens)

# Print the tagged words if they're nouns
urls = re.findall(r"(?:https?:\/\/)?\w+\.\w+\.\w+[\w\/.]+", content)
nouns = [word for word, pos in tagged if pos == "NN"]

# Top 10 nouns with frequency
print("Top 10 Nouns:")
fdist = nltk.FreqDist(nouns)
for word, frequency in fdist.most_common(10):
    print(u'{}: {}'.format(word, frequency))
#END

print("All URLs:")
for url in urls:
    print(url)
#END