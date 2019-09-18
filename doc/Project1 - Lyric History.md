1.	word analysis – over time: more word, a sign of free form of expression  
2.	Topic analysis – over time 
3.	Sentiment analysis – over time 


It's said that popular music is a reflection of society, a barometer for our collective wants, fears, and emotional states. Others are of the belief that music is more a reflection of the artist, a diary that's been flung from the nightstand drawer into the media frenzy of our modern world. In either case, music can serve as an insight into the human mind in ways that many other mediums cannot.


```python
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import pyreadr
from nltk import FreqDist
from nltk.corpus import PlaintextCorpusReader
from wordcloud import WordCloud
from sklearn.decomposition import NMF

from gensim import corpora,models
from gensim.models import LdaModel
from gensim.parsing.preprocessing import remove_stopwords,strip_punctuation, strip_numeric,strip_short

```


```python
"""
Load processed lyrics data
"""

def read_processed_lyrics() -> pd.DataFrame:
    """
    Load processed lyrics data from file
    
    return the data frame containing the data
    """
    result = pyreadr.read_r('../data/processed_lyrics.RData')
    return result['dt_lyrics']

processed_lyrics = read_processed_lyrics()
```

HTML('''<script>

code_show=true;

function code_toggle() {

if (code_show){

$('div.input').hide();

} else {

$('div.input').show();

}

code_show = !code_show

}

$( document ).ready(code_toggle);

</script>

<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')


```python

"""
no missing data
"""
processed_lyrics.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 125704 entries, 0 to 125703
    Data columns (total 7 columns):
    song            125704 non-null object
    year            125704 non-null float64
    artist          125704 non-null object
    genre           125704 non-null object
    lyrics          125704 non-null object
    id              125704 non-null int32
    stemmedwords    125704 non-null object
    dtypes: float64(1), int32(1), object(5)
    memory usage: 6.2+ MB
    


```python
processed_lyrics.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>song</th>
      <th>year</th>
      <th>artist</th>
      <th>genre</th>
      <th>lyrics</th>
      <th>id</th>
      <th>stemmedwords</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>when-you-were-with-me</td>
      <td>2009.0</td>
      <td>a</td>
      <td>Hip-Hop</td>
      <td>I stopped by the house we called our home\nIt ...</td>
      <td>1</td>
      <td>stop house call home rundown grass overgrown s...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>careless-whisper</td>
      <td>2009.0</td>
      <td>a</td>
      <td>Hip-Hop</td>
      <td>I feel so unsure\nAs I take your hand and lead...</td>
      <td>2</td>
      <td>unsure hand lead dance floor music die eyes ca...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2-59</td>
      <td>2007.0</td>
      <td>a</td>
      <td>Hip-Hop</td>
      <td>Mark:] Sunday football I got boot off the pitc...</td>
      <td>3</td>
      <td>mark sunday football boots pitch people gamble...</td>
    </tr>
    <tr>
      <td>3</td>
      <td>power-of-desire</td>
      <td>2007.0</td>
      <td>a</td>
      <td>Hip-Hop</td>
      <td>[Chris:] Fallin' for a fantasy\nI threw away m...</td>
      <td>4</td>
      <td>chris fallin fantasy threw destiny stop feelin...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>you-re-not-in-love</td>
      <td>2007.0</td>
      <td>a</td>
      <td>Hip-Hop</td>
      <td>something in the way we touch\nyou hold my han...</td>
      <td>5</td>
      <td>touch hold hand hold somethings change somethi...</td>
    </tr>
  </tbody>
</table>
</div>



# Data Exploratory 


```python
processed_lyrics.loc[processed_lyrics["year"]==112,'year']=1998
```


```python
processed_lyrics.loc[processed_lyrics['year']==702,'year'] = 2002
```


```python
def times(c):
    if c < 1980:
        return 1970
    elif c < 1990:
        return 1980
    elif c < 2000:
        return 1990
    elif c < 2010:
        return 2000
    elif c < 2020:
        return 2010
    return -1
processed_lyrics['times'] = 0
processed_lyrics['times'] = [times(x) for x in processed_lyrics['year']]
```


```python
processed_lyrics['wordcount'] = [len(x.split()) for x in processed_lyrics['stemmedwords']]
```


```python
word_avg=processed_lyrics.pivot_table(values = ['wordcount'],index=['genre'], columns=['times'], aggfunc=np.mean, fill_value=0)
```

notes:


```python
ax = plt.axes()
ax.set(title='Word frequency')
sns.heatmap(word_avg['wordcount'],fmt="d",cmap='YlGnBu',center=processed_lyrics['wordcount'].mean())
plt.title('Word frequency', fontsize=20)
plt.xlabel('time period', fontsize = 15) 
plt.ylabel('Genre', fontsize = 15) 
plt.show()
```


![png](output_14_0.png)



```python
# import readability
# def measure_readability(text):
#     result = readability.getmeasures(text,lang='en')
#     return result['readability grades']["SMOGIndex"]
```


```python
# measure_readability(processed_lyrics['stemmedwords'][1])
```


```python
from PIL import Image
hiphop_mask = np.array(Image.open('../figs/hiphop.png'))
general_mask = np.array(Image.open('../figs/singer.png'))
```


```python
from itertools import chain
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['youre','ill','ive','gotta','ya','yo','yall','uh','em','chorus','gon','shes','rhyme'])

def get_wordcloud(data):
    Words = [word.split() for word in data['stemmedwords']]
    Text = list(chain(*Words))
    Words_dis=FreqDist(Text)
    
    """
    remove stopword
    """
    for stopword in stop_words:
        if stopword in Words_dis:
            del Words_dis[stopword]
    return Words_dis
               
#     print(Words_dis.most_common(50))
```


```python
general_wc = get_wordcloud(processed_lyrics)
wordcloud = WordCloud(
        
        max_words=500,
        scale=3,
        background_color="white",  mask=general_mask,contour_width=3, contour_color='steelblue'
    )

wordcloud.generate_from_frequencies(dict(general_wc))
plt.figure(figsize=(10,15))
plt.imshow(wordcloud,cmap=plt.cm.gray, interpolation='bilinear')
plt.axis("off")
plt.show()

```


```python
hiphop_wc = get_wordcloud(processed_lyrics[processed_lyrics['genre']=='Hip-Hop'])
wordcloud = WordCloud(
        
        max_words=500,
        scale=2,
        background_color="white", mask=hiphop_mask,contour_width=3, contour_color='steelblue'
    )

wordcloud.generate_from_frequencies(dict(hiphop_wc))
plt.figure(figsize=(10,15))
plt.imshow(wordcloud,cmap=plt.cm.gray, interpolation='bilinear')
plt.axis("off")
plt.show()

fig = plot.get_figure()
```


![png](output_20_0.png)


Topic modeling 


```python

from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')

```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\rui\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    




    True




```python

vectorizer = TfidfVectorizer(stop_words=stop_words,min_df = 0.05)
```


```python
tfdif = vectorizer.fit_transform(hiphop['stemmedwords'])
nmf = NMF(n_components=4)
topic_values = nmf.fit_transform(tfdif)
```


```python
for topic_num,topic in enumerate(nmf.components_):
    message = 'topic #{} '.format(topic_num+1)
    message += ' '.join(vectorizer.get_feature_names()[i] for i in topic.argsort()[:-10 :-1])
    print(message)
```

    topic #1 time life day people live mind world rock call
    topic #2 girl baby tonight body night call shake lady time
    topic #3 niggas bitch shit ass hoes motherfucker money fuckin wit
    topic #4 love baby heart life time hurt day fall night
    


```python
topic_labels = ['life','party','anger','love']
```


```python
topic_df = pd.DataFrame(topic_values,columns=topic_labels)
```


```python
hiphop = hiphop.join(topic_df)
```


```python
hiphop.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>year</th>
      <th>id</th>
      <th>times</th>
      <th>wordcount</th>
      <th>life</th>
      <th>party</th>
      <th>anger</th>
      <th>love</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8905.000000</td>
      <td>8905.000000</td>
      <td>8905.000000</td>
      <td>8905.000000</td>
      <td>8905.000000</td>
      <td>239.000000</td>
      <td>239.000000</td>
      <td>239.000000</td>
      <td>239.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>70325.133970</td>
      <td>2007.698484</td>
      <td>70331.687254</td>
      <td>2002.350365</td>
      <td>195.834812</td>
      <td>0.034566</td>
      <td>0.017802</td>
      <td>0.028792</td>
      <td>0.015350</td>
    </tr>
    <tr>
      <th>std</th>
      <td>31302.182401</td>
      <td>3.889960</td>
      <td>31305.340041</td>
      <td>4.944740</td>
      <td>100.166441</td>
      <td>0.016707</td>
      <td>0.028162</td>
      <td>0.034010</td>
      <td>0.027373</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1989.000000</td>
      <td>1.000000</td>
      <td>1980.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>48472.000000</td>
      <td>2006.000000</td>
      <td>48476.000000</td>
      <td>2000.000000</td>
      <td>124.000000</td>
      <td>0.022555</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>71247.000000</td>
      <td>2007.000000</td>
      <td>71254.000000</td>
      <td>2000.000000</td>
      <td>190.000000</td>
      <td>0.034163</td>
      <td>0.006095</td>
      <td>0.016253</td>
      <td>0.004582</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>99855.000000</td>
      <td>2010.000000</td>
      <td>99864.000000</td>
      <td>2010.000000</td>
      <td>255.000000</td>
      <td>0.046298</td>
      <td>0.021719</td>
      <td>0.049666</td>
      <td>0.018434</td>
    </tr>
    <tr>
      <th>max</th>
      <td>122813.000000</td>
      <td>2016.000000</td>
      <td>122824.000000</td>
      <td>2010.000000</td>
      <td>1711.000000</td>
      <td>0.076750</td>
      <td>0.144898</td>
      <td>0.138383</td>
      <td>0.208335</td>
    </tr>
  </tbody>
</table>
</div>




```python
hiphop.loc[hiphop['life']>=0.05,'life']=1
hiphop.loc[hiphop['party']>=0.05,'party']=1
hiphop.loc[hiphop['anger']>=0.05,'anger']=1
hiphop.loc[hiphop['love']>=0.05,'love']=1
hiphop.loc[hiphop['life']<0.05,'life']=0
hiphop.loc[hiphop['party']<0.05,'party']=0
hiphop.loc[hiphop['anger']<0.05,'anger']=0
hiphop.loc[hiphop['love']<0.05,'love']=0
```


```python
year_topics = hiphop.groupby('year').sum().reset_index()
```


```python
plt.figure(figsize=(10,10))
plt.plot(year_topics['year'],year_topics['life'],label='life')
plt.plot(year_topics['year'],year_topics['love'],label='love')
plt.plot(year_topics['year'],year_topics['anger'],label='anger')
plt.plot(year_topics['year'],year_topics['party'],label='party')
plt.legend()
```




    <matplotlib.legend.Legend at 0x2490aadf4e0>




![png](output_32_1.png)



```python
year_topics[topic_labels].as_matrix()
sns.heatmap(year_topics[topic_labels],fmt="d",cmap='YlGnBu')
plt.show()
```

    C:\Users\rui\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      """Entry point for launching an IPython kernel.
    


![png](output_33_1.png)


old school / new school 
representative 

