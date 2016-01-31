
# coding: utf-8

# # Document retrieval from wikipedia data

# ## Fire up GraphLab Create

# In[1]:

import graphlab


# # Load some text data - from wikipedia, pages on people

# In[2]:

people = graphlab.SFrame('people_wiki.gl/')


# Data contains:  link to wikipedia article, name of person, text of article.

# In[3]:

people.head()


# In[4]:

len(people)


# # Explore the dataset and checkout the text it contains
# 
# ## Exploring the entry for president Obama

# In[5]:

obama = people[people['name'] == 'Barack Obama']


# In[6]:

obama


# In[7]:

obama['text']


# ## Exploring the entry for actor George Clooney

# In[8]:

clooney = people[people['name'] == 'George Clooney']
clooney['text']


# # Get the word counts for Obama article

# In[9]:

obama['word_count'] = graphlab.text_analytics.count_words(obama['text'])


# In[10]:

print obama['word_count']


# ## Sort the word counts for the Obama article

# ### Turning dictonary of word counts into a table

# In[11]:

obama_word_count_table = obama[['word_count']].stack('word_count', new_column_name = ['word','count'])


# ### Sorting the word counts to show most common words at the top

# In[12]:

obama_word_count_table.head()


# In[13]:

obama_word_count_table.sort('count',ascending=False)


# Most common words include uninformative words like "the", "in", "and",...

# # Compute TF-IDF for the corpus 
# 
# To give more weight to informative words, we weigh them by their TF-IDF scores.

# In[14]:

people['word_count'] = graphlab.text_analytics.count_words(people['text'])
people.head()


# In[15]:

tfidf = graphlab.text_analytics.tf_idf(people['word_count'])

# Earlier versions of GraphLab Create returned an SFrame rather than a single SArray
# This notebook was created using Graphlab Create version 1.7.1
if graphlab.version <= '1.6.1':
    tfidf = tfidf['docs']

tfidf


# In[16]:

people['tfidf'] = tfidf


# ## Examine the TF-IDF for the Obama article

# In[17]:

obama = people[people['name'] == 'Barack Obama']


# In[18]:

obama[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)


# Words with highest TF-IDF are much more informative.

# # Manually compute distances between a few people
# 
# Let's manually compare the distances between the articles for a few famous people.  

# In[19]:

clinton = people[people['name'] == 'Bill Clinton']


# In[20]:

beckham = people[people['name'] == 'David Beckham']


# ## Is Obama closer to Clinton than to Beckham?
# 
# We will use cosine distance, which is given by
# 
# (1-cosine_similarity) 
# 
# and find that the article about president Obama is closer to the one about former president Clinton than that of footballer David Beckham.

# In[21]:

graphlab.distances.cosine(obama['tfidf'][0],clinton['tfidf'][0])


# In[22]:

graphlab.distances.cosine(obama['tfidf'][0],beckham['tfidf'][0])


# # Build a nearest neighbor model for document retrieval
# 
# We now create a nearest-neighbors model and apply it to document retrieval.  

# In[23]:

knn_model = graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name')


# # Applying the nearest-neighbors model for retrieval

# ## Who is closest to Obama?

# In[24]:

knn_model.query(obama)


# As we can see, president Obama's article is closest to the one about his vice-president Biden, and those of other politicians.  

# ## Other examples of document retrieval

# In[25]:

swift = people[people['name'] == 'Taylor Swift']


# In[26]:

knn_model.query(swift)


# In[27]:

jolie = people[people['name'] == 'Angelina Jolie']


# In[28]:

knn_model.query(jolie)


# In[29]:

arnold = people[people['name'] == 'Arnold Schwarzenegger']


# In[30]:

knn_model.query(arnold)


# In[31]:

elton = people[people['name'] == 'Elton John']


# In[32]:

elton.head()


# In[33]:

elton_word_count_table = elton[['word_count']].stack('word_count', new_column_name = ['word','count'])


# In[34]:

elton_word_count_table.sort('count',ascending=False)


# In[35]:

elton_tfidf_count_table = elton[['tfidf']].stack('tfidf', new_column_name = ['word','count'])


# In[36]:

elton_tfidf_count_table.sort('count', ascending=False)


# In[37]:

victoria = people[people['name'] == 'Victoria Beckham']


# In[38]:

paul = people[people['name'] == 'Paul McCartney']


# In[39]:

#Manually compute distance


# In[40]:

graphlab.distances.cosine(elton['tfidf'][0],victoria['tfidf'][0])


# In[41]:

graphlab.distances.cosine(elton['tfidf'][0],paul['tfidf'][0])


# In[42]:

knn_model_tfidf = graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name',distance='cosine')
knn_model_word_count = graphlab.nearest_neighbors.create(people,features=['word_count'],label='name',distance='cosine')


# In[45]:

#K-Nearest Neighbor to compute the cosine distance


# In[53]:

#The distance of being closer means it is more similar


# In[52]:

knn_model_word_count.query(elton)


# In[47]:

knn_model_tfidf.query(elton)


# In[48]:

knn_model_word_count.query(victoria)


# In[49]:

knn_model_tfidf.query(victoria)

