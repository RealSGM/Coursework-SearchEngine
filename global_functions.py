'''
Student ID: 100390438, Ace Shyjan
'''

import re
import os
from math import log10 as log
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stops = set(stopwords.words('english'))

use_sublinear = True


def print_padded(text):
    print("\n" + text + "\n")

def print_line():
    print("-" * 104)

def loading_bar(count,max_count):
    max_bars = 50
    progress = int(count / max_count * max_bars)
    
    print("[" + "-" * progress + " " * (max_bars - progress) + "]", end="\r")

def os_clear():
    os.system('cls' if os.name == 'nt' else 'clear')

# Clean input
def clean_input(query):
    no_slashes = re.sub(r'[/-]', '  ', query) # Removes slashes and dashes, replaces with spaces
    no_punctuation = re.sub(r'[^\w\s]', '', no_slashes) # Removes punctuation
    re_cleaned = re.sub(r'\b\d+\b', '', no_punctuation) # Regex to remove numbers 
    tokens = word_tokenize(re_cleaned)  # Tokenize text
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stops] # Remove stop words and lemmatize and make lowercase
    return tokens

# Create bigrams list
def create_bigrams_list(cleaned_tokens):
    bigrams_list = ngrams(cleaned_tokens, 2) # Get bigrams
    return [' '.join(b) for b in bigrams_list] # Combine the bigrams into a string 

# Calculate tf-idf for a term in a document
def calculate_idf(doc_ids, vocabulary, postings, term):
    return log(len(doc_ids) / len(postings[vocabulary[term]]))

# Calculate tf for a term in a document
def calculate_tf(doc_ids, vocabulary, postings, term, file_path):
    doc = doc_ids[file_path]

    tf = postings[vocabulary[term]][doc["doc_id"]]['weight'] / doc['total_terms']
    # if use_sublinear:
        # return max(1 + log(tf), 0)
    return tf
