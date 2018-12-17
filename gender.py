
# coding: utf-8

# # Gender Identification from text

# - Akmal

# Import library

# In[1]:


import os
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

PATH = os.getcwd()


# Read data

# In[2]:


csv_path = os.path.join(PATH, "data.csv")    # Loading data csv kepada dataframe
df = pd.read_csv(csv_path, header=None, names = ["name", "chat"])

csv_path2 = os.path.join(PATH, "data2.csv")  # Loading data csv kepada dataframe
df2 = pd.read_csv(csv_path2, header=None, names = ["nama", "gender"])

df.head()


# preprocess chat

# In[3]:


for i in range(len(df)):
    chat = df.loc[i, "chat"]        # Mengubah(memastikan) data pada chat menjadi string
    df.loc[i, "chat"] = str(chat)


# Bag of word (occurence)

# In[4]:


df = df[df.chat != '[Photo]']
df = df[df.chat != '[Contact]']
df = df[df.chat != '[Sticker]']
df = df[df.chat != '[File]']         # Menghilangkan chat yang berisi tidak diinginkan seperti photo atau sticker
df = df[df.chat != 'nan']

intab = ".,/?;:[]''{}_-=+@"
outtab = "                 "        # Menghilangkan tanda baca pada chat
trantab = str.maketrans(intab, outtab)

for index, row in df.iterrows():
    row.chat = row.chat.translate(trantab)
           
df = df.reset_index(drop = True)

for i in range (0,len(df)):
    for j in range(0,len(df2)):         # Mengubah nama menjadi gender
        if df.name[i] == df2.nama[j]:
            df.name[i] = df2.gender[j]
        


# In[5]:


df.head()


# Seperate training and testing data

# In[6]:


# buang 1000 laki-laki
df = df[1000:]

df = df.sample(frac=1, random_state=10).reset_index(drop=True)  # Shuffle data pada dataframe
df.head()

df_train = df[:4000]        # Load data kepada datafram training dan test
df_test = df[4000:]


# ### Build x & y for training data

# In[7]:


count_vect = CountVectorizer(binary=True,  min_df = 10)  # Menentukan frekuensi dari kata yang muncul
x_train = count_vect.fit_transform(df_train["chat"])

vocab_ = count_vect.vocabulary_      # Vocabulary dari data

tmp = ["" for _ in range(len(vocab_))]
for word, idx in vocab_.items():
    tmp[idx] = word            
           


# In[8]:


x_train = pd.DataFrame(x_train.toarray())  # Data untuk training 
x_train.columns = tmp


# In[9]:


x_train.head()


# In[10]:


y_train = df_train.name  # y_train merupakan gender dari data 


# In[11]:


print(x_train.shape)   # Menunjukkan ukuran dari data training
print(y_train.shape)


# ### Build x & y for testing data

# In[12]:


x_test = count_vect.transform(df_test["chat"])
y_test = df_test.name                           


# In[13]:


print(x_test.shape)         # Menunjukkan ukuran data testing
print(y_test.shape)


# ## Training

# In[14]:


clf = BernoulliNB(fit_prior=False)
clf.fit(x_train, y_train)               # Training data dengan Bernoulli Naive bAYES


# ## Test

# In[15]:


pred = clf.predict(x_test)  # Prediction menggunakan data x_test


# In[16]:


print(accuracy_score(y_test, pred))  # Menunjukkan akurasi dengan membandingkan hasil Prediksi dan Data asli


# In[17]:


print(classification_report(y_test, pred))  # Menunjukkan Precission dan recall dari program


# In[18]:


conf = confusion_matrix(y_test, pred, labels=["L", "P"])  # Menunjukkan confusion matrix dari hasil
conf


# In[19]:


text = "semangat ya"
x = count_vect.transform(np.array([text]))    # Pengetesan manual terhadap model yang dilakukan
p = clf.predict(x)
print(p)

