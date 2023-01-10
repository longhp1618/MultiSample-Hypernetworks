"""This code is referenced from kaggle 
https://www.kaggle.com/code/mlwhiz/multiclass-text-classification-pytorch """


import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import torch
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re

maxlen = 750 # max number of words in a question to use
max_features = 120000 # how many unique words to use (i.e num rows in embedding vector)
batch_size = 512 # how many samples to process at once

def load_glove(word_index, path):
    EMBEDDING_FILE = path + '/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:300]
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
    
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.005838499,0.48782197
    embed_size = all_embs.shape[1]

    nb_words = min(max_features, len(word_index)+1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = embeddings_index.get(word.capitalize())
            if embedding_vector is not None: 
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def dataset(path):
    data1 = pd.read_csv(path+"drugsComTrain_raw.csv")

    data2 = pd.read_csv(path+"drugsComTest_raw.csv")

    data = pd.concat([data1,data2])[['review', 'condition', 'rating']]

    # remove NULL Values from data
    data = data[pd.notnull(data['review'])]

    data['len'] = data['review'].apply(lambda s : len(s))
    count_df = data[['condition','review']].groupby('condition').aggregate({'review':'count'}).reset_index().sort_values('review',ascending=False)
    target_conditions = count_df[count_df['review']>270]['condition'].values

    def condition_parser(x):
        if x in target_conditions:
            return x
        else:
            return "OTHER"
        
    data['condition'] = data['condition'].apply(lambda x: condition_parser(x))
    data = data[data['condition']!='OTHER']

    

    def clean_text(x):
        pattern = r'[^a-zA-z0-9\s]'
        text = re.sub(pattern, '', x)
        return x

    def clean_numbers(x):
        if bool(re.search(r'\d', x)):
            x = re.sub('[0-9]{5,}', '#####', x)
            x = re.sub('[0-9]{4}', '####', x)
            x = re.sub('[0-9]{3}', '###', x)
            x = re.sub('[0-9]{2}', '##', x)
        return x


    contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
    def _get_contractions(contraction_dict):
        contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
        return contraction_dict, contraction_re
    contractions, contractions_re = _get_contractions(contraction_dict)
    def replace_contractions(text):
        def replace(match):
            return contractions[match.group(0)]
        return contractions_re.sub(replace, text)


    data["review"] = data["review"].apply(lambda x: x.lower())

    # Clean the text
    data["review"] = data["review"].apply(lambda x: clean_text(x))

    # Clean numbers
    data["review"] = data["review"].apply(lambda x: clean_numbers(x))

    # Clean Contractions
    data["review"] = data["review"].apply(lambda x: replace_contractions(x))


    data['rating'] /= 2 #convert range of rating to [0, 5]


    from sklearn.model_selection import train_test_split
    train_X, test_X, train_Y, test_Y = train_test_split(data['review'], data[['rating', 'condition']],
                                                        test_size=0.25)
    train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y,
                                                        test_size=0.15)

    le = LabelEncoder()
    train_Y = np.concatenate((train_Y['rating'].values[:, np.newaxis], le.fit_transform(train_Y['condition'].values)[:, np.newaxis]), axis=1)
    val_Y = np.concatenate((val_Y['rating'].values[:, np.newaxis], le.transform(val_Y['condition'])[:, np.newaxis]), axis=1)
    test_Y = np.concatenate((test_Y['rating'].values[:, np.newaxis], le.transform(test_Y['condition'].values)[:, np.newaxis]), axis=1)

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    val_X = tokenizer.texts_to_sequences(val_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    ## Pad the sentences 
    train_X = pad_sequences(train_X, maxlen=maxlen)
    val_X = pad_sequences(val_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)


    x_train = torch.from_numpy(train_X).long()
    y_train = torch.from_numpy(train_Y).float()
    x_val = torch.from_numpy(val_X).long()
    y_val = torch.from_numpy(val_Y).float()
    x_test = torch.from_numpy(test_X).long()
    y_test = torch.from_numpy(test_Y).float()

    # Create Torch datasets
    train = torch.utils.data.TensorDataset(x_train, y_train)
    valid = torch.utils.data.TensorDataset(x_val, y_val)
    test = torch.utils.data.TensorDataset(x_test, y_test)
    # Create Data Loaders
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
 
    embedding_matrix = load_glove(tokenizer.word_index, path)
    return train_loader, valid_loader, test_loader, embedding_matrix