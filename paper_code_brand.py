# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 14:06:51 2021

@author: Julian van Erk
"""

#%% Libraries

import numpy as np
import pandas as pd
import json
import re
import pickle as pkl
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.optimize import fsolve
from itertools import combinations
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt

#%% functions

# functions
def multireplace(data, in_text, out_text):
    """
    Purpose: replace in_text by out_text (in order of the strings)
    
    input:
        data     dataframe column (pd.Series) or a simple string on which to apply the replacement
        in_text  strings to be replaced
        out_text replacements of the strings
    
    Return
        data_out   data after replacements
    
    """
    
    #check if input length is output lenght
    iReplacements = len(in_text)
    if iReplacements != len(out_text):
        print("the number of input texts that needs to be replaced should be the same as the number of output texts")
        return None
    
    #apply function
    data_out = data
    for i in range(iReplacements):
        if type(data) == pd.Series:
            data_out = data_out.apply(lambda column: column.replace(in_text[i], out_text[i]))
        else:
            data_out = data_out.replace(in_text[i], out_text[i])

    #return
    return data_out

def get_numumeric_part(text):
    """
    Purpose: get numeric part of a string
    
    input:
        text string we need to extract the numeric part from
    
    Return
        numeric_part  string with numeric part of the original string
    
    """
    
    numeric_part = (''.join(char for char in text if char.isdigit() or char == '.'))
    return numeric_part

def create_signature_matrix(tfidf, row_fraction = 0.5, rng = None, seed = 1234):
    """
    Purpose: create signature matrix from tf-idf matrix
    
    input:
        tfidf             tfidf matrix/dataframe
        row_fraction      fraction of the number of rows that we use to signature instances
    
    Return
        signature_matrix  signature matrix with row_fraction*#rows instances and for every 
                          column a signature.
    """
    iR, iC = tfidf.shape
    iK = round(row_fraction*iR)
    
    #initialize signature matrix
    signature_matrix = np.zeros((iK, iC))
    
    #set random number generator if not supplied
    if rng is None:
        rng = np.random.default_rng(seed)
    
    for i in range(iK):

        #create new random hyperplane 
        random_vector = 2*rng.integers(low=0, high=2, size=iR)-1     
        signature_matrix[i,:] = np.sign(random_vector[:, np.newaxis].T * tfidf)
            
    return signature_matrix

def get_r_and_b(r,b,n,t):
    """
    Purpose: helper function to calculate r and b from n and t
    
    input:
        r current estimate of number of rows 
        b current estimate of number of bands
        n number of rows in the signature matrix
        t predefined threshold
    
    Return
        equation1  difference between n and r*b
        equation2  difference between t and (1\b)^(1\r)
    """
        
    equation_1 = n - r*b
    equation_2 = t - (1/b)**(1/r)
    return(equation_1, equation_2)

#works better
def get_r_and_b2(r,b,n,t):
    """
    Purpose: helper function to calculate r and b from n and t
    
    input:
        r current estimate of number of rows 
        b current estimate of number of bands
        n number of rows in the signature matrix
        t predefined threshold
    
    Return
        equation1  difference between n and r*b
        equation2  difference between t and (1\b)^(1\r)
    """
        
    equation_1 = n + (np.log(b)*b / np.log(t))
    equation_2 = r - (n/b)
    return(equation_1, equation_2)

def hash_function(lValues):
    """
    Purpose: get a hash value for a list of +1, -1
    
    input:
        lvalues  list of +1,-1 values

    Return
        hash_key the hash_key related to this list of values
    """
    # change -1 to o
    temp_list = [max(0,int(row_elem)) for row_elem in lValues] 
    
    #change to string
    hash_key = "".join([str(elem) for elem in temp_list])
    
    #return key
    return hash_key

def lsh(signature_matrix, threshold):
    """
    Purpose: get candidate pairs from signature matrix using locality sensitive hashing
    
    input:
        signature_matrix   The signature matrix to apply lsh on
        threshold          Threshold to determine the number of rows, bands
    
    Return
        candidate_pair_matrix   N * N matrix 
        
    """
    #get n and t
    n, p = signature_matrix.shape
    t = threshold
    
    # create function specific for this signature matrix and threshold
    get_r_and_b_lhs = lambda r_and_b: get_r_and_b2(r_and_b[0],r_and_b[1],n,t) 

    #solve for r and b
    r, b = fsolve(get_r_and_b_lhs, (25,25), maxfev=10000)

    # r and be should be integers, so we are only close to the threshold
    r = int(np.floor(r))
    b = int(np.floor(b))
    
    #create hash matrix using a 3D array
    hash_matrix = np.zeros((b, r, p)) #first element is depth, second rows, third columns
    
    #fill hash matrix
    for band in range(b):
        hash_matrix[band,:,:] = signature_matrix[(band*r):((band+1)*r),:]
        
    #get hash keys (the identification of the buckets a product is put in)
    hash_key_matrix = np.zeros((b,p), dtype = f"|S{r}") #dtype is string of max r char
    for band in range(b):
        for product in range(p):
            hash_key_matrix[band, product] = hash_function(hash_matrix[band, :, product])
            
    #now we put the products into buckets: dictionary with hash_key as key and list of indices of products as values
    bucket_list = [None]*b
    candidate_pair_matrix = np.zeros((p,p))
    for band in range(b):
        #create dictionary for band band
        band_buckets = {}
        #fill buckets
        for product in range(p):
            #get key 
            key = hash_key_matrix[band, product]
            #check if we already have an element in this bucket
            if key in band_buckets:
                #get all other products in this bucket
                for other_product in band_buckets[key]:
                    #set candidate pair matrix elements of all combis in the list with this element to 1.
                    candidate_pair_matrix[other_product, product] = 1
                    candidate_pair_matrix[product, other_product] = 1
                #add element to bucket
                band_buckets[key] += [product]
            else:
                #create new bucket with key as bucketidentificaton and the product index as value
                band_buckets[key] = [product]
        
        #add dictionary of bucktes to bucket_list
        bucket_list[band] = band_buckets
  
    return candidate_pair_matrix
    
def feature_creation(x1, x2):
    """
    Purpose:
        Create features for product pair (1,2) to use in the SVM classifier
        
    input:   
        x1    tfidf representation product 1
        x2    tfidf representation product 2
    
    Return
        output   feature values for product pair (1,2)
    """
    #get full elements

    
    # get where 1,2 are zero
    x1_zero = np.where(x1 == 0, 0, 1)
    x2_zero = np.where(x2 == 0, 0, 1)

    output = (1-x1_zero)*x2_zero*-x1 + (1-x2_zero)*x1_zero*-x2 + x1_zero*x2_zero*np.sqrt(x1*x2)
    
    return output
    


#%% load data

# get JSON file
with open('C:/Users/Rob/Documents/computerscience/TVs-all-merged.json') as file:
    data = json.load(file)

#create a file with non nested elements
data_non_nested = []
for product in data.values(): 
    #loop over all instances of a product
    for product_instance in product:     
        #add product instance to list
        data_non_nested.append(product_instance)
        
# transform to dataframe
df_data = pd.DataFrame(data_non_nested)

#%% Data preprocessing

# transform title, all keys and values to lowercase
df_data['featuresMap'] = df_data['featuresMap'].apply(lambda column: dict((key.lower(),value.lower()) for key, value in column.items()))
df_data['title'] = df_data['title'].apply(lambda column: column.lower())

#remove "-", "!", "(", ")", "/' in title  (not in featuresmap)
df_data['title'] = multireplace(df_data['title'], ["-", '–',"!","(",")","/", "[", "]"], 8*[""])

#apply normalization on both the title and featuresMap
original_representations = [" inches", "inches", ' "','"',' ”','”', ' -inch', '-inch', ' inch',
                            " -hz", "-hz", " hertz", "hertz", " hz",
                            " lbs.", " lbs", "lbs.", " pounds", "pounds", " pound", "pound",  " lb ", "lb ", "lb.", " lb.",
                            "led-lcd"]
normalized_representations= 9*['inch']+5*['hz'] + 11*['lbs']+['ledlcd']
df_data['title'] = multireplace(df_data['title'] , original_representations, normalized_representations)
df_data['featuresMap'] = df_data['featuresMap'].apply(lambda column: 
                                                      dict((key, multireplace(value,  
                                                                              original_representations, 
                                                                              normalized_representations)) 
                                                      for key, value in column.items()))

    
    
    
    
#%% model words detection

#regex for title model words
regex_title_model_words = "([azA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*)"
regex_value_model_words = "(^\d+(\.\d+)?[a-zA-Z]+$|^\d+(\.\d+)?$)"
regex_numeric_part = r"\d+(\.\d+)?$"

# make list of title model words (using the regex) 
nobs =  df_data.shape[0]
title_model_words = [] 

for title in df_data['title']:
    
    #find title model words
    regex_model_words_output = re.findall(regex_title_model_words, title)
    title_model_words += [regex_model_words_output[i][0] for i in range(len(regex_model_words_output))]

#get unique elements
title_model_words = list(set(title_model_words))

# add brands and product type as title_model_words
brands = ["lg","philips","samsung","toshiba","nec","supersonic","sharp","viewsonic", "sony", "coby","sansui",
          "tcl","hisense","optoma","haier","panasonic","rca","naxa","vizio","magnavox", "sanyo", "insignia",
          'jvc','proscan','venturer','westinghouse','sceptre']     
product_type = ["led", "plasma","lcd"]
title_model_words = title_model_words + brands + product_type

#make list of value model words
value_model_words = []

for featuresDict in df_data['featuresMap']:
    for value in featuresDict.values():
        #find value model words
        regex_model_words_output = re.findall(regex_value_model_words, value)
        temp_list = [regex_model_words_output[i][0] for i in range(len(regex_model_words_output))]
        
        #remove non-numerical part
        temp_list = [get_numumeric_part(text) for text in temp_list]
        value_model_words += temp_list
    
#get unique elements
value_model_words = list(set(value_model_words))    

# remove value model words that are already title words, as we check the title words
# for both the title and the values (and we don't want to count double)
value_model_words = list(set(value_model_words) - (set(title_model_words) & set(value_model_words)))

#get all model words
model_words = title_model_words + value_model_words







#%% Create count matrix

#create count matrix
n_value_words = len(value_model_words)
n_title_words = len(title_model_words)
nwords = n_value_words + n_title_words #disjoint sets
count_matrix = np.zeros((nwords,nobs))

#fill count matrix
for j in range(nobs):
    print(j)
    
    #get title, values
    temp_title = df_data['title'].iloc[j]
    temp_dict = df_data['featuresMap'].iloc[j]
    
    #check if title/ value pairs have title model words
    for i in range(n_title_words):
        
        #get title word and regex
        word = re.escape(model_words[i]) #escape non-numerical parts (.,:, etc.)
        temp_regex = rf"(?<![\w\-\.\:\–\/])\b{word}\b(?![\w\-\.\:\–\/])" #note the rf instead of r
        
        #check if title word is in title
        count_matrix[i,j] += len(re.findall(temp_regex, temp_title))
        
        #check if title word is in values
        for temp_value in temp_dict.values():
            count_matrix[i,j] += len(re.findall(temp_regex, temp_value))     
            
    #check if value pairs have value model words
    for i in range(n_title_words, nwords):
        #get title word and regex
        word = re.escape(model_words[i]) #escape non-numerical parts (.,:, etc.)
        temp_regex = rf"(?<![\w\-\.\:\–\/])\b{word}(?![0-9\-\.\:\–\/])" #note the rf instead of r
        
        #check if title word is in values
        for temp_value in temp_dict.values():
              
            count_matrix[i,j] += len(re.findall(temp_regex, temp_value))    
        
        




            
#%% get product brand information
brands = ["lg","philips","samsung","toshiba","supersonic","sharp","viewsonic", "sony","nec","coby","sansui",
          "tcl","hisense","optoma","haier","panasonic","rca","naxa","vizio","magnavox", "sanyo", "insignia",
          'jvc','proscan','venturer','westinghouse','sceptre']     
df_count = pd.DataFrame(count_matrix, index=model_words)    
brand_count = df_count.loc[brands]    
products = brand_count.shape[1]
df_data["brand"] = products*[['unknown']]

#loop over all products to fill brand names
for j in range(brand_count.shape[1]):
    column = brand_count.loc[:,j]
    temp_brand = [column[column >0].index.values]
    df_data.loc[j,"brand"] = [[el for el in temp_brand]]
    


#%% filtering of count matrix

#get indices of each news site
bestbuy_ind = df_data.index[df_data['shop'] == "bestbuy.com"].tolist()
amazon_ind = df_data.index[df_data['shop'] == "amazon.com"].tolist()
newegg_ind = df_data.index[df_data['shop'] == "newegg.com"].tolist()
thenerds_ind = df_data.index[df_data['shop'] == "thenerds.net"].tolist()

#get number of words per site
words_per_site = np.array([np.sum(count_matrix[:,bestbuy_ind], axis = 1),
                    np.sum(count_matrix[:,amazon_ind], axis = 1),
                    np.sum(count_matrix[:,newegg_ind], axis = 1),
                    np.sum(count_matrix[:,thenerds_ind], axis = 1)])

#get indices where there are at least 2 nonzero sites and word in max 50 percent of documents
two_sites_nonzero_ind = np.where(np.count_nonzero(words_per_site, axis = 0) > 1)[0]
max_50perc_ind = np.where(np.count_nonzero(count_matrix, axis = 1) < nobs/2)[0]
filter_ind = list(set(two_sites_nonzero_ind) & set(max_50perc_ind))
filter_ind.sort()

#get new count matrix and new words
count_matrix = count_matrix[filter_ind,:]
model_words = [model_words[i] for i in filter_ind]

#define dataframe
df_count = pd.DataFrame(count_matrix, index=model_words, columns = df_data['shop']+" "+df_data['modelID'])

# save count matrix 
with open('C:/Users/Rob/Documents/computerscience/count_matrix_brand.pkl', 'wb') as file:
    pkl.dump(count_matrix, file, protocol=pkl.HIGHEST_PROTOCOL)

with open('C:/Users/Rob/Documents/computerscience/df_count_brand.pkl', 'wb') as file:
    pkl.dump(df_count, file, protocol=pkl.HIGHEST_PROTOCOL)







#%% get count matrices

with open('C:/Users/Rob/Documents/computerscience/count_matrix_brand.pkl', 'rb') as file:
    count_matrix= pkl.load(file)
    
with open('C:/Users/Rob/Documents/computerscience/df_count_brand.pkl', 'rb') as file:
    df_count = pkl.load(file)








#%% get tf-idf representation

#define transformer
tf_idf_transf = TfidfTransformer()

#transform
tfidf = tf_idf_transf.fit_transform(X = df_count)
df_tfidf = pd.DataFrame(tfidf.todense(), index=model_words, columns = df_data['shop']+" "+df_data['modelID'])

#save representations
with open('C:/Users/Rob/Documents/computerscience/tfidf_brand.pkl', 'wb') as file:
    pkl.dump(tfidf, file, protocol=pkl.HIGHEST_PROTOCOL)


    
  
    
  
#%%    
# Part 2 of the project     
    


#%% load represenations
with open('C:/Users/Rob/Documents/computerscience/tfidf_brand.pkl', 'rb') as file:
    tfidf = pkl.load(file)

with open('C:/Users/Rob/Documents/computerscience/df_tfidf_brand.pkl', 'rb') as file:
    df_tfidf = pkl.load(file)

with open('C:/Users/Rob/Documents/computerscience/count_matrix_brand.pkl', 'rb') as file:
    count_matrix= pkl.load(file)
    
with open('C:/Users/Rob/Documents/computerscience/df_count_brand.pkl', 'rb') as file:
    df_count = pkl.load(file)

  
    
    
    

#%% get the duplicate matrix
p = df_data.shape[0]
duplicate_matrix = np.zeros((p,p))
for i in range(p - 1):
    for j in range(i, p):
        #get product id
        product_i = df_data['modelID'].iloc[i]
        product_j = df_data['modelID'].iloc[j]
        
        #check if the same
        if product_i == product_j:
            duplicate_matrix[i,j] = 1
            duplicate_matrix[j,i] = 1
            
#%% get same website matrix

same_website_matrix = np.zeros((p,p))
for i in range(p ):
    for j in range(i, p):
        #get product id
        shop_i = df_data['shop'].iloc[i]
        shop_j = df_data['shop'].iloc[j]
        
        #check if the same
        if shop_i == shop_j:
            same_website_matrix[i,j] = 1
            same_website_matrix[j,i] = 1
            
#%% get other brand matrix           
other_brand_matrix = np.ones((p,p))
for i in range(p ):
    for j in range(i, p):
        #get brand
        brand_i = list(df_data['brand'].iloc[i][0])
        brand_j = list(df_data['brand'].iloc[j][0])
        
        #check if the same
        if brand_i == [] or brand_j == [] or len(set(brand_i).intersection(set(brand_j))) > 0:
            other_brand_matrix[i,j] = 0
            other_brand_matrix[j,i] = 0

    
#%% get all information for bootstrapping later
p = df_data.shape[0]
n_features = tfidf.shape[0]
n_same_website = int(np.sum(np.triu(same_website_matrix, k = 1)))
#n_other_brand = int(np.sum(np.triu( other_brand_matrix, k = 1)))
classification_data = np.empty((int(0.5*p*(p-1) - n_same_website),5 + n_features))
counter = 0
for i in range(p - 1):
    for j in range(i+1, p):

        #if same shop, we do not need the data
        if same_website_matrix[i,j] == 1:
            continue
        
        #if ohter brand, we do not need the data
        if other_brand_matrix[i,j] == 1:
            continue
        
        if counter % 1000 == 0:
            print(counter)
            
        #check if the same product
        if duplicate_matrix[i,j] ==1:
            same_product = 1
        else:
            same_product = 0
            
        #get tfidf representation
        tfidf_i = tfidf[:,i]
        tfidf_j = tfidf[:,j]
            
        #get cosine similarity
        cosine_sim = np.dot(tfidf_i.T, tfidf_j).toarray()[0,0]
        
        #get other features
        a = np.array(tfidf_i.todense()).reshape(n_features,)
        b = np.array(tfidf_j.todense()).reshape(n_features,)
        features = feature_creation(a, b)   

        #store values and update counter
        classification_data[counter,0:5] = [i,j,same_product, cosine_sim, cosine_sim**2]
        classification_data[counter,5:] = features
        counter += 1
      


classification_data = classification_data[0:counter, :]
df = pd.DataFrame(classification_data[:,2:], index=[classification_data[:,0],classification_data[:,1]], 
                  columns = ["same_product", "cosine_sim","cosine_sim^2"] + list(df_tfidf.index))
df = df.astype(pd.SparseDtype("float", 0.0)) #sparse matrix: 6.4 GB vs 27.1 MB

#save data
with open('C:/Users/Rob/Documents/computerscience/df_brand.pkl', 'wb') as file:
    pkl.dump(df, file, protocol=pkl.HIGHEST_PROTOCOL)
  



#%% 
#load data
with open('C:/Users/Rob/Documents/computerscience/df_brand.pkl', 'rb') as file:
    df = pkl.load(file)


#%% bootstrapping 
df = df.iloc[:, 0:359]#remove value words
p = df_data.shape[0]
n_features = tfidf.shape[0]

#set number of bootstraps
bootstraps = 20

#create random number generator
seed = 1234
rng = np.random.default_rng(seed)

#set product indices
product_ind = np.arange(p)

#set thresholds values
thresholds = [0.90,0.875,0.85,0.825,0.80,0.775,0.75,0.725,0.70,0.675,0.65,0.60,0.55,0.525,0.50,0.475,0.45,0.30]
n_thresholds = len(thresholds)

#initialize performance measures
Df_lsh = np.zeros((n_thresholds,bootstraps), dtype = int)
Df_svm = np.zeros((n_thresholds,bootstraps), dtype = int)
Nc =  np.zeros((n_thresholds,bootstraps), dtype = int)
Dn =  np.zeros((n_thresholds,bootstraps), dtype = int)
svm_tot_pos = np.zeros((n_thresholds,bootstraps), dtype = int)
test_size = np.zeros((n_thresholds,bootstraps), dtype = int)

#predifine svm and grid search 
model = make_pipeline(StandardScaler(with_mean=False, with_std=True),
                      SGDClassifier(random_state=0, loss = "hinge", penalty = "l2",
                                tol = 1e-4, max_iter= 10000, warm_start= True, n_jobs = -1,
                                n_iter_no_change = 100))
param_grid = {'sgdclassifier__alpha': list(np.exp(np.linspace(np.log(10**(-9)),np.log(10**(-0)),10))), 
            'sgdclassifier__class_weight': [{1: (x)/(1+x), 0:1/(1+x)} for x in [1.0,3.0,5.0,10.0,15.0,20.0,50.0]] } 

#define grid search
clf = GridSearchCV(model, param_grid, n_jobs = -1, scoring = 'f1', cv = 5)

#start
told = time.time()
for b in range(bootstraps):
    tnew = time.time()
    print(f"Starting with bootstrap {b+1} time = {round((tnew-told)/60,2)}")
    
    #get bootstrapped samples
    train_ind = np.sort(rng.choice(product_ind, size = p, replace = True))
    test_ind = np.sort(np.array(list(set(product_ind) - set(train_ind))))
    
    #get all combinations of train that have not the same webshop
    train_comb_total = combinations(train_ind, r = 2)
    train_comb = []
    for i,j in train_comb_total:     
        #we do not check observations from the same webshop
        if same_website_matrix[i,j] == 1 or other_brand_matrix[i,j] == 1:
            continue
        else:
            train_comb.append((i,j))
        
    #get all combinations of test that have not the same webshop
    test_comb_total = combinations(test_ind, r = 2)
    test_comb = []
    for i,j in test_comb_total:
        
        #we do not check observations from the same webshop
        if same_website_matrix[i,j] == 1 or other_brand_matrix[i,j] == 1:
            continue
        else:
            test_comb.append((i,j))
    
    #define train, test
    df_train = df.loc[train_comb]
    df_test = df.loc[test_comb]
    
    # get number of duplicates in test sample
    Dn[:,b] = np.sum(df_test.iloc[:,0])
    test_size[:,b] = df_test.shape[0]
   
    for j in range(len(thresholds)):     
        t = thresholds[j]
        print(f"t = {t}, b = {b+1}")
        
        #create signature matrix train set
        train_tfidf = tfidf[:, train_ind]
        train_signature = create_signature_matrix(train_tfidf, rng = rng)
        
        #create signature matrix test set
        test_tfidf = tfidf[:, test_ind]
        test_signature = create_signature_matrix(test_tfidf, rng = rng)
               
        #apply lsh on train set
        train_lsh_candidate_matrix = lsh(train_signature, threshold = t)
        
        #get candidate pairs
        candidate_pairs_train = np.transpose(np.nonzero(np.triu(train_lsh_candidate_matrix, k = 1)))
        candidate_pairs_train = [(train_ind[el[0]], train_ind[el[1]]) for el in candidate_pairs_train]
        candidate_pairs_train = pd.MultiIndex.from_frame(pd.DataFrame(data = candidate_pairs_train))
        candidate_pairs_train = candidate_pairs_train.intersection(df_train.index)
    
        #set temp train set
        df_train_candidate = df_train.loc[candidate_pairs_train]
        
        #set x, y
        y_train, X_train = df_train_candidate.iloc[:,0], df_train_candidate.iloc[:,1:]
    
        #train model 
        if len(candidate_pairs_train) > 0 and np.sum(y_train) >= 5: #we need at least 5 train positive samples
            print(f"Start training at {round((time.time()- told)/60, 2)}")
            clf.fit(X_train, y_train)
            print(f"Params: {clf.best_params_}")
            print(f"Score: {clf.best_score_}")
            print(f"stop training at {round((time.time()- told)/60, 2)}")
    
        ########
        ## testing
        
        #apply lsh on test set
        test_lsh_candidate_matrix = lsh(test_signature, threshold = t)
    
        #get candidate pairs
        candidate_pairs_test = np.transpose(np.nonzero(np.triu(test_lsh_candidate_matrix, k = 1)))
        candidate_pairs_test = [(test_ind[el[0]], test_ind[el[1]]) for el in candidate_pairs_test]
        candidate_pairs_test = pd.MultiIndex.from_frame(pd.DataFrame(data = candidate_pairs_test))
        candidate_pairs_test = candidate_pairs_test.intersection(df_test.index)
    
        #set temp test set
        df_test_candidate = df_test.loc[candidate_pairs_test]
    
        #set x, y
        y_test, X_test = df_test_candidate.iloc[:,0], df_test_candidate.iloc[:,1:]
        
        #get the number of comparisons made
        Nc[j,b] = len(candidate_pairs_test)
        
        #get performance lsh
        Df_lsh[j,b] = np.sum(y_test)
        
        #get performance model
        if len(candidate_pairs_test) > 0 and np.sum(y_train) >= 5:
            y_predict = clf.predict(X_test)
            #y_predict = model.predict(X_test)
            svm_tot_pos[j,b] = np.sum(y_predict)
            Df_svm[j,b] = np.sum(np.logical_and(y_test, y_predict))

            

#save
result_list_brand = [Dn, Nc, Df_lsh, Df_svm, svm_tot_pos, test_size]

#save data
with open('C:/Users/Rob/Documents/computerscience/result_list_brand.pkl', 'wb') as file:
    pkl.dump(result_list_brand, file, protocol=pkl.HIGHEST_PROTOCOL)

#%% calculate summary stats.
#load data
with open('C:/Users/Rob/Documents/computerscience/result_list_brand.pkl', 'rb') as file:
    result_list = pkl.load(file)
   
# get statistics bootstrapping    
Dn = result_list[0]
#Dn = Df_lsh[-1,:]
Nc = result_list[1]
Df_lsh = result_list[2]
Df_svm = result_list[3]
svm_tot_pos = result_list[4]
test_size = result_list[5]
#test_size = Nc[-1,:]

#calculate lsh performance measures
frac_of_comp = np.mean(Nc/test_size, axis = 1) 
PQ_lsh =np.true_divide(Df_lsh, Nc)
PQ_lsh_mean = np.nanmean(PQ_lsh , axis = 1)
PC_lsh = Df_lsh/ Dn 
PC_lsh_mean = np.mean( PC_lsh , axis = 1)
F1_lsh = 2*PQ_lsh*PC_lsh /(PQ_lsh + PC_lsh)
F1_lsh_mean = np.nanmean(F1_lsh, axis = 1)

#calculate svm performance measures
PQ_svm =  Df_svm / svm_tot_pos 
PQ_svm_mean =  np.nanmean(PQ_svm , axis = 1)
PC_svm = Df_svm / Dn
PC_svm_mean = np.mean(PC_svm  , axis = 1)
F1_svm = 2*PQ_svm*PC_svm /(PQ_svm + PC_svm)
F1_svm_mean = np.nanmean(F1_svm, axis = 1)


#%% make figures

fig, ax = plt.subplots( dpi=300)
ax.plot(frac_of_comp, PQ_lsh_mean, 'k')
ax.set_xlabel("Fraction of comparisons")
ax.set_ylabel("Pair quality")
plt.xlim(0,0.2)
plt.ylim(0,0.2)
plt.grid(True)
plt.show()


fig, ax = plt.subplots( dpi=300)
ax.plot(frac_of_comp, PC_lsh_mean, 'k')
ax.set_xlabel("Fraction of comparisons")
ax.set_ylabel("Pair completeness")
plt.ylim(0,1)
plt.grid(True)
plt.show()


fig, ax = plt.subplots( dpi=300)
ax.plot(frac_of_comp, F1_lsh_mean, 'k')
ax.set_xlabel("Fraction of comparisons")
ax.set_ylabel("F1 score")
plt.xlim(0,0.2)
plt.grid(True)
plt.show()

fig, ax = plt.subplots( dpi=300)
ax.plot(frac_of_comp, F1_svm_mean, 'k', label = "SVMP")
ax.plot( frac_of_comp, n_thresholds*[0.525], 'k--', label = "MSM")
ax.set_xlabel("Fraction of comparisons")
ax.set_ylabel("F1 score")
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(frac_of_comp, PQ_svm_mean)
plt.xscale('linear')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(frac_of_comp, PC_svm_mean)
plt.xscale('linear')
plt.grid(True)
plt.show()
