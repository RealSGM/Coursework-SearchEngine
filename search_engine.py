'''
Student ID: 100390438, Ace Shyjan
'''

# Imports ----------------------------------------------------------------------

# Main Import
from nltk.corpus import wordnet
from nltk import pos_tag
from textdistance import jaro_winkler
import math
from global_functions import *

# Other Imports
from time import time as timestamp
import time
import json

# Global Variables -------------------------------------------------------------
postings = {}
doc_ids = {}
vocabulary = {}
document_vectors = {}
summations = {}

query = ''
query_terms = []
file_query_string = ''
print_to_console = False

# Functions --------------------------------------------------------------------

# Get file path from doc_id
def get_file_path(doc_id):
    for file_path in doc_ids.keys():
        if str(doc_ids[file_path]['doc_id']) == str(doc_id):
            return file_path
    return ''

def get_all_docs_with_term(term):
    return [get_file_path(doc_id) for doc_id in postings[vocabulary[term]]]

# Spell checker ----------------------------------------------------------------
# Loop through query terms and replace words with closest word in vocabulary
# Use Jaro-Winkler similarity to calculate similarity between query term and vocabulary term
def spell_checker(query_terms):
    cleaned_query_terms = []
    for query_term in query_terms:
        if query_term in vocabulary:
            cleaned_query_terms.append(query_term)
        else:
            closest_word = ''
            max_similarity = -1
            threshold = 0.9
        
            for term in vocabulary:
                similarity = jaro_winkler(query_term, term) # Calculate similarity between query term and vocabulary term
                if similarity >= threshold and similarity > max_similarity: # If similarity is above threshold, replace query term with vocabulary term
                    closest_word = term
                    max_similarity = similarity

            if closest_word != '':
                print(f'Using "{closest_word}" instead of "{query_term}"')
                cleaned_query_terms.append(closest_word)
    return cleaned_query_terms

# Query Expansion --------------------------------------------------------------
# Loop through query terms and find related synonyms
# Ensure that synyonms are actually related to the query term
def get_synonyms(tokens):
    named_entities = pos_tag(tokens)
    synonyms_list = []
    
    for named_entity in named_entities:
        synsets = wordnet.synsets(named_entity[0])
        for syn in synsets:
            for lemma in syn.lemmas():
                synonym = lemma.name()
                if synonym != named_entity[0]: 
                    if pos_tag([synonym])[0][1] == named_entity[1]:
                        if synonym in vocabulary and synonym not in synonyms_list:
                            synonyms_list.append(synonym)
    return synonyms_list

# Bigrams ----------------------------------------------------------------------
# Create bigrams from query terms
# Add cleaned bigrams to query terms if in vocabulary
def add_bigrams_to_query_terms(query_terms):
    new_bigrams = create_bigrams_list(query_terms)
    returning_bigrams = []
    for bigram in new_bigrams:
        if bigram in vocabulary:
            returning_bigrams.append(bigram)
    return returning_bigrams

# Calculate tf-idfs for query   terms
def calculate_tf_idfs(query_terms): # Calculate tf-idfs for query terms
    tf_idfs = {}
    for query_term in query_terms:
        vocab_id = vocabulary[query_term]
        postings_table = postings[vocab_id]
        idf = calculate_idf(doc_ids, vocabulary, postings, query_term) # Calculate idf for query term
        
        for doc_id in postings_table.keys(): # Loop through all postings for that postings table
            tf = calculate_tf(doc_ids, vocabulary, postings, query_term, get_file_path(doc_id))
            if doc_id not in tf_idfs:
                tf_idfs[doc_id] = tf * idf
            else:
                tf_idfs[doc_id] += tf * idf
                
    return tf_idfs

# Vector Space Model -----------------------------------------------------------
# Calculate query vector
# Loop through all terms in vocabulary and calculate idf for each term in query
def calculate_query_vector(query_terms):
    return [calculate_idf(doc_ids, vocabulary, postings, term) if term in query_terms else 0 for term in vocabulary]

# Calculate cosine similarity
# Loop through all documents and calculate cosine similarity for each document
# Cosine Similarity = (A . B) / (||A|| * ||B||)
# A = query vector, B = document vector
def calculate_cosine_similarity(query_vector, document_vectors):
    cosine_similarities = {}
    for doc_id, doc_vector in document_vectors.items():
        dot_product = sum(query_value * doc_value for query_value, doc_value in zip(query_vector, doc_vector)) # Calculate dot product
        query_magnitude = math.sqrt(sum(value ** 2 for value in query_vector)) 
        doc_magnitude = math.sqrt(sum(value ** 2 for value in doc_vector)) 
        
        similarity = dot_product / (query_magnitude * doc_magnitude + 1e-9) 
        cosine_similarities[doc_id] = similarity

    return cosine_similarities

def get_summation_scores(query_terms,file_path):
    summation_scores = {}
    for query_term in query_terms:
        for summation in summations[file_path]:
            cleaned_summation = clean_input(summations[file_path][summation])
            if query_term in cleaned_summation:
                if summation not in summation_scores:
                    summation_scores[summation] = 1
                else:
                    summation_scores[summation] += 1
    return summation_scores

# Output tf-idfs / cosine similarities for query terms
def output_ranked_results(results_dict,time,type):
    global file_query_string
    results_list = sorted(results_dict.items(), key=lambda x: x[1], reverse=True) # Sort results by value
    results_list = [doc for doc in results_list if doc[1] > 0] # Remove results with 0 value
        
    if len(results_list) > 10:
        results_list = results_list[:10] # Only show top 10 results
        
    print(f'{type} Results:')
    print(f'Query: {query}')
    print(f'Time taken: {time} seconds\n')
    file_query_string += f'{type},{query},{time}\n-\n'
    
    if results_dict == {}:
        print('No results found')
    else:
        for i, result in enumerate(results_list):
            file_path = get_file_path(result[0])
            summation_scores = get_summation_scores(query_terms, file_path)
            
            print(f"Rank #{i + 1}: {file_path}, Score: {result[1]}")
            file_query_string += f'{file_path},{result[1]}\n'
            
            if summation_scores:
                key_max = max(summation_scores, key=summation_scores.get)
                print(f"Summation: {summations[file_path].get(key_max, '')}")
            else:
                print("No summations found for this document")
            
            print('')
    file_query_string += '-\n'
    
def process_query(query): # Process query
    global query_terms
    
    query_terms = clean_input(query) 
    cleaned_query_terms = spell_checker(query_terms) 
    
    queried_bigrams = add_bigrams_to_query_terms(cleaned_query_terms) 
    cleaned_query_terms.extend(queried_bigrams)
    # expanded_query = get_synonyms(cleaned_query_terms)
    # cleaned_query_terms.extend(expanded_query)

    return cleaned_query_terms
    
def ranked_retrieval(cleaned_query_terms, type): # Ranked retrieval
    start_time = time.time()
    results = {}
    if type == "TF-IDF":
        results = calculate_tf_idfs(cleaned_query_terms)
    elif type == "Cosine-Similarity":
        query_vector = calculate_query_vector(cleaned_query_terms)
        results = calculate_cosine_similarity(query_vector, document_vectors)
    end_time = time.time()
    output_ranked_results(results, end_time - start_time, type)
    print_line()
    
def load_files(): # Load files from disk
    global doc_ids, vocabulary, postings, document_vectors, summations
    with open('data/doc_ids.txt', 'r') as f: # Load doc_ids from file
        doc_ids = json.load(f)
        doc_ids = {file_path: {'doc_id': int(doc_ids[file_path]['doc_id']), 
                            'total_terms': int(doc_ids[file_path]['total_terms'])} for file_path in doc_ids}

    with open('data/vocabulary.txt', 'r') as f: # Load vocabulary from file
        vocabulary = json.load(f)
        vocabulary = {term: int(vocabulary[term]) for term in vocabulary} # Convert keys to integers
        
    with open('data/postings.txt', 'r') as f: # Load postings from file
        postings = json.load(f)
        
    with open('data/document_vectors.txt', 'r') as f: # Load document_vectors from file
        document_vectors = json.load(f)
        document_vectors = {doc_id: [float(value) for value in document_vectors[doc_id]] for doc_id in document_vectors}

    with open('data/summations.txt', 'r') as f: # Load summations from file
        summations = json.load(f)

    # Convert keys to integers and convert values to integers and floats
    postings = {
        int(term_id_str): {
            int(doc_id_str): {
                'term_freq': int(postings[term_id_str][doc_id_str]['term_freq']),
                'weight': float(postings[term_id_str][doc_id_str]['weight'])
            } for doc_id_str in postings[term_id_str]
        } for term_id_str in postings
    }
    return doc_ids, vocabulary, postings, document_vectors, summations

def main_loop(): # Main loop 
    global query, query_terms
    while query != 'N':
        os_clear()
        query = input('Please enter your query (Enter "N" to quit): ') # Get query from user
        print("\n")
        
        if query != 'N': 
            
            cleaned_query_terms = process_query(query)
            ranked_retrieval(cleaned_query_terms, "TF-IDF")
            ranked_retrieval(cleaned_query_terms, "Cosine-Similarity")
            
            input('Press enter to continue...')
            
    t = str(int(timestamp())) # Get current date and time
    file_name = f'{t}.txt'
    with open(f'results/{file_name}', 'w') as f:
        f.write(file_query_string)
    
    
# Main ------------------------------------------------------------------------- 
if __name__ == '__main__': 
    if not os.path.exists('results'):
        os.makedirs('results')
    load_files()
    main_loop()

    # search_queries = [
    #     'Action-Adventure Games',
    #     'RPG Games for PlayStation',
    #     'The Guy Game',
    #     'Spyder-Man',
    #     'star-wars-battlefront',
    #     'Iron Man',
    #     'James Earl Cash',
    #     'Crazy Taxi',
    #     'James Bond',
    #     'The Lord of the Rings: The Two Towers (PS2)'
    # ]
    # for search_query in search_queries:
    #     print_padded(f"Search Query: {search_query}")
        
    #     query = search_query
    #     cleaned_query_terms = process_query(search_query)
    #     ranked_retrieval(cleaned_query_terms, "TF-IDF")
    #     ranked_retrieval(cleaned_query_terms, "Cosine-Similarity")
        
    # t = str(int(timestamp())) # Get current date and time
    # file_name = f'{t}.txt'
    # with open(f'results/{file_name}', 'w') as f:
    #     f.write(file_query_string)
