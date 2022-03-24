# CS395 Homework 3: Bag of Words Model
# In this model, a text is represented as a bag (multiset) of its words,
# disregarding grammar and word order but keeping multiplicity

import numpy as np
import pprint as pp
import string

def load_text_from_file(file_location):
    """Loads text file from file_location (filename or path)""" 
    raw_lines = []
    with open(file_location) as f:
        for line in f:
            if line.strip():
                #print(line)
                raw_lines.append(line.strip())
    raw_lines = " ".join(raw_lines)
    return raw_lines

def preprocess_text_data(s):
    """Removes punctuation marks. Also sets text to lowercase."""
    cleaned = s
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
    cleaned = cleaned.lower()
    
    return cleaned

def construct_model_dictionary(text_corpus):
    """Construct a list of all unique tokens that 
        exist in the corpus (across all documents in it).
        NB: Assume that tokens are delimited by 
        spaces only. Beware of empty/null tokens 
        (remove them if they exist). Sort the 
        dictionary alphabetically (in ascending order).
        This will return a list of 
        unique tokens (return type is list, not dict)."""
    
    # construct a list of unique tokens from text_corpus
    # initialize empty list
    unique_tokens_list = []
    # for each key in our text_corpus, split text into string tokens and save them as a set unique_tokens
    for key in text_corpus:
        tokens = text_corpus[key].split()
        unique_tokens = set(tokens)
        # for each unique string token in the new set, append to our unique_tokens_list
        for token in unique_tokens:
            unique_tokens_list.append(token)
    
    # sorting the list of unique tokens (unique_tokens_list)
    unique_tokens_list.sort()
    
    # removing any duplicate or empty tokens and saving final token list as model_dict
    model_dict =  list(dict.fromkeys(unique_tokens_list))

    return model_dict

def score_document(input_doc, model_dictionary):
    """Using the input model_dictionary as your reference, 
    construct a vector representation of the input document. 
    The resulting vector will be as long as the model_dictionary. 
    It will encode the presence of each model_dictionary token 
    with the value 1--or 0 if absent. NB: This will yield 
    a binary vector, not a count of frequencies."""
    
    # initialize input_doc_words to be a list of split() words from the input_doc
    input_doc_words = input_doc.split()
    # initialize features to the input model_dictionary
    features = model_dictionary
    # construct a vector representation of the input document
    doc_vector = []
    
    # for each token word in features, if the word exists in the input_doc_words
    # add a 1 to doc_vector, else add a 0
    for token in features:
        if token in input_doc_words:
            # storing result in doc_vector
            doc_vector.append(1)
        else:
            doc_vector.append(0)
            
    return doc_vector

def find_distinguishing_tokens(doc_vectors, model_dictionary):
    """From the document vectors you constructed, 
        identify/list all tokens that appear in 
        the gatsby document but NOT in the alice document.
        Also identify/list all tokens that appear in the 
        alice document but NOT in the gatsby document.
        Store the result in the distinguishers dictionary (type dict)."""
    
    # retreiving document info
    gatsby_doc = np.array(doc_vectors['the_great_gatsby_excerpt'])
    alice_doc = np.array(doc_vectors['alice_in_wonderland_excerpt'])

    # Gatsby Array > Alice Array
    # if value of gatsby_doc is greater than alice_doc, word token is unique and only exists in gatsby
    # create distribution vector where gatsby_dist[i] = True when word token is unique
    gatsby_dist = np.greater(gatsby_doc, alice_doc)
    
    ind = 0 # index value
    gatsby_unique = [] # vector to hold unique word tokens in gatsby
    # cycle through each token in model_dictionary, which contains total list of token words
    for token in model_dictionary:
        if gatsby_dist[ind] == True:
            # if current index of gatsby_dist = True, word token in gatsby and not alice
            # append token to gatsby_unique
            gatsby_unique.append(token)
            # increment ind to match current token index
            ind += 1
        else:
            # even if current index of gatsby_unique is false,
            # we still want to increment ind to match current token index
            ind += 1


    # Alice Array > Gatsby Array
    # if value of alice_doc is greater than gatsby_doc, word token is unique and only exists in alice
    # create distribution vector where alice_dist[i] = True when word token is unique
    alice_dist = np.greater(alice_doc, gatsby_doc)

    ind2 = 0 # index value
    alice_unique = [] # vector to hold unique word tokens in alice
    # cycle through each token in model_dictionary, which contains total list of token words
    for token in model_dictionary:
        if alice_dist[ind2] == True:
            # if current index of alice_dist = True, word token in alice and not gatsby
            # append token to alice_unique
            alice_unique.append(token)
            # increment ind2 to match current token index
            ind2 += 1
        else:
            # even if current index of alice_unique is false,
            # we still want to increment ind2 to match current token index
            ind2 += 1
    
    distinguishers = {} # dict to hold unique word token values for gatsby and alice
    # inputting resulting data into final schema
    distinguishers["IN_GATSBY_not_in_alice"] = gatsby_unique
    distinguishers["IN_ALICE_not_in_gatsby"] = alice_unique
    
    return distinguishers

# Two input files excerpted from the following sources:
# 1. "The Great Gatsby" (1925) by F. Scott Fitzgerald.
# 2. "Alice's Adventures in Wonderland" (1865) by Lewis Carroll.
# Central goal: Construct a basic bag-of-words model over the given corpus
input_files = ["the_great_gatsby_excerpt.txt", "alice_in_wonderland_excerpt.txt"]

text_corpus = {}
# load and clean input files
for elem in input_files:
    text_data = load_text_from_file(elem)
    cleaned_text = preprocess_text_data(text_data)
    # peek inside
    print("after cleaning {}:\n{}...MORE TEXT...\n".format(elem, cleaned_text[:80]))
    text_corpus[elem.replace(".txt", "")] = cleaned_text

# construct model dictionary over the entire corpus
model_dictionary = construct_model_dictionary(text_corpus)

# construct document vectors
doc_vectors = {}
for label, cleaned_text in text_corpus.items():
    # compute the vector representation of cleaned_text (the current document)
    current_vec = score_document(cleaned_text, model_dictionary)
    # store that vector in the doc_vectors object
    doc_vectors[label] = current_vec

# find the tokens that appear in a document but not the other
distinguishers = find_distinguishing_tokens(doc_vectors, model_dictionary)
for label, v in distinguishers.items():
    print("tokens {} -> count = {};\nselected examples = {}\n".format(label, len(v), v[:5]))