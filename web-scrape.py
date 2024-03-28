'''
Student ID: 100390438, Ace Shyjan
'''

'''
Corpus format: {file_path: [{(term, term_freq): weight} ... {(term, term_freq): weight}]}
Doc ids format: {file_path: {doc_id: doc_id, total_terms: total_terms}}
Vocabulary format: {term: term_id}
Postings Format: {term_id: {doc_id: {term_freq: term_freq, weight: weight}, doc_id: {term_freq: term_freq, weight: weight}
Document Vectors Format: {doc_id: [weight, weight, weight, weight, ... weight]}
'''

# Imports ----------------------------------------------------------------------
import os
import json
from bs4 import BeautifulSoup
from collections import Counter
from global_functions import *

# Global Variables -------------------------------------------------------------
desired_tags = {
    'title': 10,
    'p': 3,
    'td': 5,
    'div': 3 
}

desired_summations = ['p','div']

corpus = {}
doc_ids = {}
vocabulary = {}
postings = {}
summations = {}

folder_path = 'videogames' # Folder path
files_in_folder = os.listdir(folder_path) # Get all files in folder

summation_id = 0


# Functions --------------------------------------------------------------------

# Cleans and tokenizes the content of a given HTML tag.
def clean_and_tokenize(tag, file_path):
    global summation_id
    if tag.name in desired_summations:
        i_tag = tag.find('i')
        if i_tag:
            i_tag.unwrap()
        for inner_tag in tag.find_all(recursive=True):
            inner_tag.decompose()  # Remove inner tags 
        
        content = tag.get_text()
        content = ' '.join(content.split())  
        content = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', content) 
        
        for line in content:
            summations.setdefault(file_path, {})[str(summation_id)] = line  
            summation_id += 1 
    
    content = tag.get_text()
    cleaned_tokens = clean_input(content)
    cleaned_bigrams_list = create_bigrams_list(cleaned_tokens)
    return Counter(cleaned_tokens), Counter(cleaned_bigrams_list)

# Tokenizes the HTML content and calculates the weighted term frequencies for desired tags.
def tokenizer(htmlContent, file_path): 
    soup = BeautifulSoup(htmlContent, 'html.parser')
    weighted_term_freq = []
    
    for desired_tag in desired_tags: # Find all tags with desired_tags
        tags = soup.find_all(desired_tag)
        if desired_tag == 'div':
            tags = soup.find_all(desired_tag, {"id": "content"})
        elif desired_tag == 'td':
            tags = soup.find_all(desired_tag, {"class": "gameBioInfoText"})
    
        for tag in tags:
            term_freq, bigrams_freq = clean_and_tokenize(tag, file_path) # Clean and tokenize tag, include bigrams for all tags
            weighted_term_freq.extend([{term: desired_tags[desired_tag]} for term in term_freq.items()]) # Add weighted term frequencies to list
            weighted_term_freq.extend([{bigram: desired_tags[desired_tag] * 4} for bigram in bigrams_freq.items()])
    return weighted_term_freq

# Add doc_id to doc_ids
def add_to_doc_ids(file_path,doc_id): 
    if file_path not in doc_ids: # Check if doc_id is in doc_ids
        total_terms = 0
        for term in corpus[file_path]: # Calculate total terms
            total_terms += list(term.keys())[0][1]
            
        doc_ids[file_path] = {'doc_id': doc_id, 'total_terms': total_terms} # Add doc_id to doc_ids
        doc_id += 1
    return doc_id

# Adds a term to the postings and vocabulary dictionaries.
def add_to_postings_and_vocab(file_path,term,term_freq,weight,term_id):
    vocabulary[term] = term_id
    postings[vocabulary[term]] = {doc_ids[file_path]['doc_id']: {'term_freq': term_freq, 'weight': weight}} # Add term to postings
    term_id += 1
    return term_id

# Check if the document ID is present in the postings for a given term.
def check_doc_id_in_postings(file_path, term, term_freq, weight):
    for doc in postings[vocabulary[term]].keys():
        if doc == doc_ids[file_path]['doc_id']:
            postings[vocabulary[term]][doc]['term_freq'] += term_freq
            postings[vocabulary[term]][doc]['weight'] += weight
            return True
    return False

# Scans a document corpus and updates the vocabulary and postings with the terms and their weights.
def scan_document_corpus(file_path, term_id):
    for weighted_token in corpus[file_path]: # Add terms to vocabulary and postings
        token = list(weighted_token.keys())[0]
        term = token[0]
        term_freq = token[1]
        weight = weighted_token[token]
        
        if term not in vocabulary: 
            term_id = add_to_postings_and_vocab(file_path,term,term_freq,weight,term_id)
        elif not check_doc_id_in_postings(file_path,term,term_freq,weight): # Check if doc_id is in postings[term_id]
            postings[vocabulary[term]][doc_ids[file_path]['doc_id']] = {'term_freq': term_freq, 'weight': weight} # Add doc_id to postings[term_id]
    return term_id

# Calculates the document vectors for each document in the collection.
def calculate_document_vectors():
    document_vectors = {}
    for i in range(len(doc_ids)):
        file_path = list(doc_ids.keys())[i]
        doc_id = doc_ids[file_path]['doc_id']
        doc_vector = []
        for term in vocabulary:
            if vocabulary[term] in postings and doc_id in postings[vocabulary[term]]:
                tf = calculate_tf(doc_ids, vocabulary, postings, term, file_path)
                idf = calculate_idf(doc_ids, vocabulary, postings, term)
                doc_vector.append(tf * idf)
            else:
                doc_vector.append(0)
        document_vectors[doc_id] = doc_vector
        loading_bar(i + 1,len(doc_ids))
    return document_vectors

# Save the vocabulary, doc_ids, postings, document_vectors, and summations to separate files.
def save_files(document_vectors):
    if not os.path.exists('data'):
        os.makedirs('data')

    for i, data in enumerate([vocabulary, doc_ids, postings, document_vectors, summations]):
        loading_bar(i, 5)
        with open(f"data/{['vocabulary', 'doc_ids', 'postings', 'document_vectors', 'summations'][i]}.txt", "w") as f:
            json.dump(data, f)
    loading_bar(5, 5)
    
# Scrape web pages and perform various operations on the corpus.
def web_scraping_algorithm():
    term_id = 0
    doc_id = 0
    
    for i, file_path in enumerate(files_in_folder):
        if file_path.endswith('.html'):
            full_path = os.path.join(folder_path, file_path)
            with open(full_path, 'r') as f:
                content = f.read()
                
                # Tokenize file content and update IDs
                corpus[file_path] = tokenizer(content, file_path)
                doc_id = add_to_doc_ids(file_path, doc_id)
                term_id = scan_document_corpus(file_path, term_id)
        
        loading_bar(i + 1, len(files_in_folder)) # Update loading bar

# Main -------------------------------------------------------------------------
if __name__ == '__main__':
    os_clear()

    print("Scanning corpus...")
    web_scraping_algorithm() # Scrape web pages and perform various operations on the corpus.
    print_padded("Finished scanning corpus.")
        
    print("Calculating document vectors...")
    document_vectors = calculate_document_vectors() # Calculate the document vectors for each document in the collection.
    print_padded("Finished calculating document vectors.")

    print("Saving files...")
    save_files(document_vectors) # Save the vocabulary, doc_ids, postings, document_vectors, and summations to separate files.
    print_padded("Finished saving files.")