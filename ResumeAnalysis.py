#!/usr/bin/env python
# coding: utf-8

# In[57]:


# Import all libraries
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import pickle
import re
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import string
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import warnings


# In[58]:


warnings.filterwarnings('ignore')
np.set_printoptions(precision=4)

nltk.download('stopwords')
nltk.download('punkt')


# In[59]:


# Load dataset
#data = pd.read_csv('/content/UpdatedResumeDataSet.csv', engine='python')
data = pd.read_csv('UpdatedResumeDataSet.csv' ,encoding='utf-8') # Comment this line and uncomment the above line if this does not work for you
data.head()


# In[60]:


data.info()


# In[61]:


print("Verisetinde yer alan özgeçmişlerin kategorilerinin gösterimi: \n\n")
print(data['Category'].unique())


# In[62]:


# Print unique categories of resumes
print(data['Category'].value_counts())


# In[63]:


# Drop rows where category is "Testing" and store new size of dataset
data = data[data.Category != 'Testing']
data_size = len(data)


# In[64]:


import seaborn as sns
plt.figure(figsize=(20,5))
plt.xticks(rotation=90)
ax=sns.countplot(x="Category", data=data)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
plt.grid()


# In[65]:


# Bar graph visualization
plt.figure(figsize=(15,15))
plt.xticks(rotation=90)
sns.countplot(y="Category", data=data)


# In[66]:


from matplotlib.gridspec import GridSpec
targetCounts = data['Category'].value_counts()
targetLabels  = data['Category'].unique()

plt.figure(1, figsize=(22,22))
the_grid = GridSpec(2, 2)


cmap = plt.get_cmap('coolwarm')
plt.subplot(the_grid[0, 1], aspect=1, title='CATEGORY DISTRIBUTION')

source_pie = plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True)
plt.show()


# In[67]:


# Get set of stopwords
from nltk.corpus import stopwords
stopwords_set = set(stopwords.words('english')+['``',"''"])


# In[68]:


# Function to clean resume text
def clean_text(resume_text):
    resume_text = re.sub('http\S+\s*', ' ', resume_text)  # remove URLs
    resume_text = re.sub('RT|cc', ' ', resume_text)  # remove RT and cc
    resume_text = re.sub('#\S+', '', resume_text)  # remove hashtags
    resume_text = re.sub('@\S+', '  ', resume_text)  # remove mentions
    resume_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@
    [\]^_`{|}~"""), ' ', resume_text)  # remove punctuations
    resume_text = re.sub(r'[^\x00-\x7f]',r' ', resume_text) 
    resume_text = re.sub('\s+', ' ', resume_text)  # remove extra whitespace
    resume_text = resume_text.lower()  # convert to lowercase
    resume_text_tokens = word_tokenize(resume_text)  # tokenize
    filtered_text = [w for w in resume_text_tokens if not w in stopwords_set]
    # remove stopwords
    return ' '.join(filtered_text)


# In[69]:


data["Resume"] = data["Resume"].str.lower()
data.head()


# In[70]:


PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

data["Resume"] = data["Resume"].apply(lambda text: remove_punctuation(text))
data.head()


# In[71]:


from nltk.corpus import stopwords
", ".join(stopwords.words('english'))


# In[72]:


STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

data["Resume"] = data["Resume"].apply(lambda text: remove_stopwords(text))
data.head()


# In[73]:


from collections import Counter
cnt = Counter()
for text in data["Resume"].values:
    for word in text.split():
        cnt[word] += 1
        
cnt.most_common(10)


# In[74]:


FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
def remove_freqwords(text):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])

data["Resume"] = data["Resume"].apply(lambda text: remove_freqwords(text))
data.head()


# In[75]:


n_rare_words = 10
RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])
def remove_rarewords(text):
    """custom function to remove the rare words"""
    return " ".join([word for word in str(text).split() if word not in RAREWORDS])

data["Resume"] = data["Resume"].apply(lambda text: remove_rarewords(text))
data.head()


# In[80]:


print(data['Resume'][891])


# In[81]:


from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

data["Resume"] = data["Resume"].apply(lambda text: stem_words(text))
data.head()


# In[82]:


from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

data["Resume"] = data["Resume"].apply(lambda text: lemmatize_words(text))
data.head()


# In[83]:


print(data['Resume'][891])


# In[84]:


# Print a sample original resume
print('--- Original resume ---')
print(data['Resume'][0])


# In[85]:


# Clean the resume
data['cleaned_resume'] = data.Resume.apply(lambda x: clean_text(x))


# In[86]:


print('--- Cleaned resume ---')
print(data['cleaned_resume'][0])


# In[87]:


# Get features and labels from data and shuffle
features = data['cleaned_resume'].values
original_labels = data['Category'].values
labels = original_labels[:]


# In[88]:


for i in range(data_size):
  labels[i] = str(labels[i].lower())  # convert to lowercase
  labels[i] = labels[i].replace(" ", "")  # use hyphens to convert multi-token labels into single tokens


# In[89]:


import random
random.seed(20)
features, labels = shuffle(features, labels, random_state=20)


# In[90]:


# Print example feature and label
print(features[0])
print(labels[0])


# In[91]:


# Split for train and test
train_split = 0.8
train_size = int(train_split * data_size)

train_features = features[:train_size]
train_labels = labels[:train_size]

test_features = features[train_size:]
test_labels = labels[train_size:]


# In[92]:


# Print size of each split
print(len(train_labels))
print(len(test_labels))


# In[93]:


# Tokenize feature data and print word dictionary
vocab_size = 6000
oov_tok = '<OOV>'

feature_tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
feature_tokenizer.fit_on_texts(features)

feature_index = feature_tokenizer.word_index
print(dict(list(feature_index.items())))


# In[94]:


# Print example sequences from train and test datasets
train_feature_sequences = feature_tokenizer.texts_to_sequences(train_features)
print(train_feature_sequences[0])

test_feature_sequences = feature_tokenizer.texts_to_sequences(test_features)
print(test_feature_sequences[0])


# In[95]:


# Tokenize label data and print label dictionary
label_tokenizer = Tokenizer(lower=True)
label_tokenizer.fit_on_texts(labels)

label_index = label_tokenizer.word_index
print(dict(list(label_index.items())))


# In[96]:


# Print example label encodings from train and test datasets
train_label_sequences = label_tokenizer.texts_to_sequences(train_labels)
print(train_label_sequences[0])

test_label_sequences = label_tokenizer.texts_to_sequences(test_labels)
print(test_label_sequences[0])


# In[97]:


# Pad sequences for feature data
max_length = 300
trunc_type = 'post'
pad_type = 'post'

train_feature_padded = pad_sequences(train_feature_sequences, maxlen=max_length, padding=pad_type, truncating=trunc_type)
test_feature_padded = pad_sequences(test_feature_sequences, maxlen=max_length, padding=pad_type, truncating=trunc_type)


# In[98]:


# Print example padded sequences from train and test datasets
print(train_feature_padded[0])
print(test_feature_padded[0])


# In[99]:


# Define the neural network
embedding_dim = 64

model = tf.keras.Sequential([
  # Add an Embedding layer expecting input vocab of size 6000, and output embedding dimension of size 64 we set at the top
  tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=300),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
  #tf.keras.layers.Dense(embedding_dim, activation='relu'),

  # use ReLU in place of tanh function since they are very good alternatives of each other.
  #tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(embedding_dim, activation='relu'),

  # Add a Dense layer with 25 units and softmax activation for probability distribution
  tf.keras.layers.Dense(25, activation='softmax')
])


# In[100]:


model.summary()


# In[101]:


# Alternative model
embedding_dim = 64
num_categories = 25


# In[102]:


model = tf.keras.Sequential([
  tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=300),
  tf.keras.layers.GlobalMaxPooling1D(),

  # use ReLU in place of tanh function since they are very good alternatives of each other.
  tf.keras.layers.Dense(128, activation='relu'),
  # Add a Dense layer with 25 units and softmax activation for probability distribution
  tf.keras.layers.Dense(num_categories, activation='softmax'),])

model.summary()


# In[103]:


# Compile the model and convert train/test data into NumPy arrays
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[104]:


# Features
train_feature_padded = np.array(train_feature_padded)
test_feature_padded = np.array(test_feature_padded)


# In[105]:


# Labels
train_label_sequences = np.array(train_label_sequences)
test_label_sequences = np.array(test_label_sequences)


# In[106]:


# Print example values
print(train_feature_padded[0])
print(train_label_sequences[0])
print(test_feature_padded[0])
print(test_label_sequences[0])


# In[107]:


# Train the neural network
num_epochs = 25

history = model.fit(train_feature_padded, train_label_sequences, epochs=num_epochs, shuffle = True, validation_data=(test_feature_padded, test_label_sequences), verbose=2)


# In[108]:


# Plot the training and validation loss 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[109]:


# print example feature and its correct label
print(test_features[5])
print(test_labels[5])


# In[110]:


# Create padded sequence for example
resume_example = test_features[5]
example_sequence = feature_tokenizer.texts_to_sequences([resume_example])
example_padded = pad_sequences(example_sequence, maxlen=max_length, padding=pad_type, truncating=trunc_type)
example_padded = np.array(example_padded)
print(example_padded)


# In[111]:


# Make a prediction
prediction = model.predict(example_padded)


# In[112]:


# Verify that prediction has correct format
print(prediction[0])
print(len(prediction[0]))  # should be 25
print(np.sum(prediction[0]))  # should be 1


# In[113]:


# Find maximum value in prediction and its index
print(max(prediction[0]))  # confidence in prediction (as a fraction of 1)
print(np.argmax(prediction[0])) # should be 3 which corresponds to python developer


# In[114]:


# Indices of top 5 most probable solutions
indices = np.argpartition(prediction[0], -5)[-5:]
indices = indices[np.argsort(prediction[0][indices])]
indices = list(reversed(indices))
print(indices)


# In[115]:


# Save model
model.save('model')


# In[116]:


# Save feature tokenizer
with open('feature_tokenizer.pickle', 'wb') as handle:
    pickle.dump(feature_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[117]:


# Save reverse dictionary of labels to encodings
label_to_encoding = dict(list(label_index.items()))
print(label_to_encoding)


# In[118]:


encoding_to_label = {}
for k, v in label_to_encoding.items():
  encoding_to_label[v] = k
print(encoding_to_label)


# In[119]:


with open('dictionary.pickle', 'wb') as handle:
    pickle.dump(encoding_to_label, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(encoding_to_label[np.argmax(prediction[0])])


# In[120]:


data.head()


# In[121]:


data = data[["Category","cleaned_resume"]]
data.head()


# In[122]:


review = data["cleaned_resume"]


# # Vord2Wec

# In[123]:


import os
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
from gensim.models import Word2Vec
import nltk
nltk.download('wordnet')
stemmer = SnowballStemmer('english')

from numpy import dot
from numpy.linalg import norm


# In[124]:


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            if token == 'xxxx':
                continue
            result.append(lemmatize_stemming(token))
    
    return result


# In[125]:


processed_docs = data['cleaned_resume'].map(preprocess)
processed_docs =list(processed_docs)


# In[126]:


processed_docs[:10] # clean document


# In[127]:


def word2vec_model():
    w2v_model = Word2Vec(min_count=1,
                     window=3,
                     vector_size=50,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20)
    
    w2v_model.build_vocab(processed_docs)
    w2v_model.train(processed_docs, total_examples=w2v_model.corpus_count, epochs=300, report_delay=1)
    
    return w2v_model


# In[128]:


w2v_model = word2vec_model()
#w2v_model.save('word2vec_model')


# In[129]:


emb_vec = w2v_model.wv


# In[130]:


emb_vec['program'] # It will return vector representation of the word anak


# # Finding similarity between two vector using cosine similarity

# In[131]:


def find_similarity(sen1, sen2, model):
    p_sen1 = preprocess(sen1)
    p_sen2 = preprocess(sen2)
    
    sen_vec1 = np.zeros(50)
    sen_vec2 = np.zeros(50)
    for val in p_sen1:
        sen_vec1 = np.add(sen_vec1, model[val])

    for val in p_sen2:
        sen_vec2 = np.add(sen_vec2, model[val])
    
    return dot(sen_vec1,sen_vec2)/(norm(sen_vec1)*norm(sen_vec2))


# In[132]:


find_similarity('areas interest deep learning control system', 'areas interest deep learning control system',emb_vec )


# In[133]:


find_similarity('areas interest deep learning control system', 'areas interest control system deep learning ',emb_vec )


# In[140]:


find_similarity('areas interest deep learning control system', 'try',emb_vec )


# In[141]:


df = data


# In[142]:


def sampling_dataset(df):
    count = 5000
    class_df_sampled = pd.DataFrame(columns = ["cleaned_resume","Category"])
    temp = []
    for c in df.Category.unique():
        class_indexes = df[df.Category == c].index
        random_indexes = np.random.choice(class_indexes, count, replace=True)
        temp.append(df.loc[random_indexes])
        
    for each_df in temp:
        class_df_sampled = pd.concat([class_df_sampled,each_df],axis=0)
    
    return class_df_sampled


# In[143]:


df = sampling_dataset(df)
df.reset_index(drop=True,inplace=True)
print (df.head())
print (df.shape)


# # TF-IDF

# In[147]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize


# In[148]:


# vectorizers for creating the document-term-matrix (DTM)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import TruncatedSVD


# In[149]:


#stop-words
stop_words=set(nltk.corpus.stopwords.words('english'))
vect =TfidfVectorizer(stop_words=stop_words,max_features=1000)


# In[150]:


vect_text=vect.fit_transform(data["cleaned_resume"])


# In[151]:


print(vect.get_feature_names())


# In[152]:


print(vect_text.shape)
type(vect_text)


# In[153]:


idf=vect.idf_


# In[154]:


dd=dict(zip(vect.get_feature_names(), idf))
l=sorted(dd, key=(dd).get)
# print(l)


# In[155]:


print(l[0],l[-1])
print(dd['python'])
print(dd['work']) 


# # LSA

# In[156]:


lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)


# In[157]:


print(lsa_top[0])
print(lsa_top.shape)  # (no_of_doc*no_of_topics)


# In[158]:


l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print("Topic ",i," : ",topic*100)


# In[159]:


print(lsa_model.components_.shape) # (no_of_topics*no_of_words)
print(lsa_model.components_)


# In[160]:


# most important words for each topic
vocab = vect.get_feature_names()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# # LDA 

# In[161]:


from sklearn.decomposition import LatentDirichletAllocation
lda_model=LatentDirichletAllocation(n_components=10,learning_method='online',random_state=42,max_iter=1) 


# In[162]:


lda_top=lda_model.fit_transform(vect_text)


# In[163]:


print(lda_top.shape)
print(lda_top[0])


# In[164]:


sum=0
for i in lda_top[0]:
  sum=sum+i
print(sum)


# In[165]:


# composition of doc 0 for eg
print("Document 0: ")
for i,topic in enumerate(lda_top[0]):
  print("Topic ",i,": ",topic*100,"%")


# In[166]:


print(lda_model.components_[0])
print(lda_model.components_.shape)  # (no_of_topics*no_of_words)


# In[167]:


# most important words for each topic
vocab = vect.get_feature_names()

for i, comp in enumerate(lda_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:5]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# In[168]:


from wordcloud import WordCloud
# Generate a word cloud image for given topic
def draw_word_cloud(index):
  imp_words_topic=""
  comp=lda_model.components_[index]
  vocab_comp = zip(vocab, comp)
  sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:25]
  for word in sorted_words:
    imp_words_topic=imp_words_topic+" "+word[0]    
  wordcloud = WordCloud(width=900, height=600).generate(imp_words_topic)
  plt.figure( figsize=(5,5))
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.tight_layout()
  plt.show()


# In[169]:


# topic 0
draw_word_cloud(0)


# In[170]:


# topic 1
draw_word_cloud(1)


# In[171]:


# topic 2
draw_word_cloud(2)


# In[172]:


# topic 3
draw_word_cloud(3)


# In[173]:


# topic 4
draw_word_cloud(4)


# In[174]:


# topic 5
draw_word_cloud(5)


# In[175]:


# topic 6
draw_word_cloud(6)


# In[176]:


# topic 7
draw_word_cloud(7)


# In[177]:


# topic 8
draw_word_cloud(8)


# In[178]:


# topic 9
draw_word_cloud(9)

