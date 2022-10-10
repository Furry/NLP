import os
import nltk
import json
import nltk.corpus
import eventlet
import threading
import multiprocessing
import greenlet

from nltk import word_tokenize, sent_tokenize
from nltk.util import ngrams

import nltk.corpus.reader as CorpusReader

# Download the libraries
nltk.download('stopwords')
nltk.download('punkt')

# Utility Functions
def normalize(words):
    words = [word for word in words if word not in nltk.corpus.stopwords.words('english')]
    # Remove the punctuation
    words = [word for word in words if word.isalpha()]
    # Lowercase the words
    words = [word.lower() for word in words]
    return words
#END

def getOr(obj, key, default):
    if key in obj:
        return obj[key]
    else:
        return default
    #END
#END

# Define the objects for remembering the ratio
unigram = {}
bigram = {}
trigram = {}
fourgram = {}

# Set the paths for the sets
testSetPath = os.path.abspath(os.path.join("../", 'DUC 2005 Dataset/TestSet'))
trainingSetPath = os.path.abspath(os.path.join("../", 'DUC 2005 Dataset/TrainingSet'))

# Load each corpus, seems like XML except no file extension for some reason?
testCorpus = CorpusReader.XMLCorpusReader(testSetPath, '.*\.*')
trainingCorpus = CorpusReader.XMLCorpusReader(trainingSetPath, '.*\.*')

# Get the Training Data
if os.path.exists("./trainingData.json"):
    print("Training file found! Using existing data...")
    # Load the training data
    with open("./trainingData.json", "r") as file:
        trainingData = json.load(file)
        unigram = trainingData["unigram"]
        bigram = trainingData["bigram"]
        trigram = trainingData["trigram"]
        fourgram = trainingData["fourgram"]
    #END
else:
    print("Training file not found! Generating new data...")
    count = len(trainingCorpus.fileids())
    for file in trainingCorpus.fileids():
        print("Processing file: ", file, " (", count, " remaining)")
        words = normalize(trainingCorpus.words(file))

        # Increment each unigram, bigram, trigram, and fourgram
        for word in words:
            if word in unigram:
                unigram[word] += 1
            else:
                unigram[word] = 1
            #END
        #END

        for n in list(ngrams(words, 2)):
            if n in bigram:
                bigram[' '.join(n)] += 1
            else:
                bigram[' '.join(n)] = 1
            #END
        #END

        for n in list(ngrams(words, 3)):
            if n in trigram:
                trigram[' '.join(n)] += 1
            else:
                trigram[' '.join(n)] = 1
            #END
        #END

        for n in list(ngrams(words, 4)):
            if n in fourgram:
                fourgram[' '.join(n)] += 1
            else:
                fourgram[' '.join(n)] = 1
            #END
        #END
        count -= 1
    #END

    # Save the training data
    with open("./trainingData.json", "w") as file:
        json.dump({
            "unigram": unigram,
            "bigram": bigram,
            "trigram": trigram,
            "fourgram": fourgram
        }, file)
    #END

    print("Training data saved!")
#END

print("Normalizing Unigrams...")
total = 0
for word in unigram:
    total += unigram[word]
#END

for word in unigram:
    unigram[word] = unigram[word] / total
#END
print("Done")

def calculateProbability(args):
    sentence, alpha = args
    # Normalize the sentence
    words = normalize(word_tokenize(sentence))

    unigramList = [ getOr(unigram, word, alpha) for word in words ]
    fourgramList = []
    trigramList = []
    bigramList = []

    bigramPerp = []
    trigramPerp = []
    fourgramPerp = []

    for ngram in list(ngrams(words, 2)):
        bigrams = [n for n in bigram.keys() if n.split(' ')[0] == ngram[0]]
        if len(bigrams) > 0:
            bigramList.append(getOr(bigram, ' '.join(ngram), alpha) / len(bigrams))
        else:
            bigramList.append(alpha)
        #END
    #END

    for ngram in list(ngrams(words, 3)):
        trigrams = [n for n in trigram.keys() if ' '.join(n.split(' ')[:2]) == ' '.join(ngram[:2])]
        if len(trigrams) > 0:
            trigramList.append(getOr(trigram, ' '.join(ngram), alpha) / len(trigrams))
        else:
            trigramList.append(alpha)
        #END
    #END

    for ngram in list(ngrams(words, 4)):
        fourgrams = [n for n in fourgram.keys() if ' '.join(n.split(' ')[:3]) == ' '.join(ngram[:3])]
        if len(fourgrams) > 0:
            fourgramList.append(getOr(fourgram, ' '.join(ngram), alpha) / len(fourgrams))
        else:
            fourgramList.append(alpha)
        #END
    #END

    for i in range(0, len(fourgramList)-1):
        bigramPerp.append(unigramList[i] * bigramList[i])
        trigramPerp.append(unigramList[i] * bigramList[i] * trigramList[i])
        fourgramPerp.append(unigramList[i] * bigramList[i] * trigramList[i] * fourgramList[i])
    #END

    print("+----------------------------------------+")
    print("| Sentence: ", sentence)
    print("| Unigram Average: ", sum(unigramList) / len(unigramList))
    print("| Bigram Average: ", sum(bigramPerp) / len(bigramPerp))
    print("| Trigram Average: ", sum(trigramPerp) / len(trigramPerp))
    print("| Fourgram Average: ", sum(fourgramPerp) / len(fourgramPerp))
    # Perplexity
    print("| Unigram Perplexity: ", pow(2, -sum(unigramList) / len(unigramList)))
    print("| Bigram Perplexity: ", pow(2, -sum(bigramPerp) / len(bigramPerp)))
    print("| Trigram Perplexity: ", pow(2, -sum(trigramPerp) / len(trigramPerp)))
    print("| Fourgram Perplexity: ", pow(2, -sum(fourgramPerp) / len(fourgramPerp)))
    print("+----------------------------------------+")

    return (
        unigramList, bigramPerp, trigramPerp, fourgramPerp,
        bigramPerp, trigramPerp, fourgramPerp
    )
#END

for file in testCorpus.fileids():
    print("Processing file: ", file)
    words = testCorpus.words(file)
    sents = sent_tokenize(' '.join(words))

    for sent in sents:
        calculateProbability((sent, 1))
    break
#END

