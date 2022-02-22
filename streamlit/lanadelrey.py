import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from wordcloud import WordCloud
import brewer2mpl
import nltk
from nltk.text import Text
from nltk import bigrams
from PIL import Image
import networkx as nx
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
import spacy
from spacy import displacy

foto = Image.open('lanadelrey.jpg')

st.set_page_config(
     page_title="Análise de Letras - Lana Del Rey",
     page_icon=foto,
     initial_sidebar_state="collapsed")

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #858285;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #A93EAA;
    color:#000000;
    }

div.stTextInput > div > div > input {
    color: #7e0080;
    background-color: #a1a1a1
}
</style>""", unsafe_allow_html=True)

st.set_option('deprecation.showPyplotGlobalUse', False)

plt.style.use('dark_background')

#st.title('Análise das músicas da Lana Del Rey')

html_temp_inicio = """ <div style =background-image:url('https://giffiles.alphacoders.com/180/180048.gif');padding:13px;border: 5px solid;"> 
    <h1 style ="color:#6efff6;font-family:helvetica;webkit-text-stroke: 2px black;margin-right:20px;text-align:right;">Análise de <br> Músicas da <br> Lana Del Rey</h1> 
    </div> """
st.markdown(html_temp_inicio, unsafe_allow_html = True) 


st.markdown('---')
st.markdown('# Sobre a Cantora')
st.image(foto, caption="Lana na capa do seu mais recente álbum, 'Blue Banisters'", width = 690)
st.markdown('''Lana Del Rey é uma cantora norte-americana de Indie Pop, que carrega influências do Rock, Hip Hop, Jazz, Blues e, mais recentemente, até Country. Seu primeiro álbum digital, lançado em 2010, deu início à carreira que mantém até hoje, totalizando 8 albuns gravados.

Ao passar dos anos, a estética da cantora foi adquirindo diferentes formas, resultado de mudanças em sua vida pessoal e forma de enxergar o universo. Tais mudanças, portanto, são notáveis em cada álbum.

O intuito desta amostragem é, utilizando técnicas de Processamento de Linguagem Natural, Machine Learning e Data Science, explicitar algumas de tais alterações na arte da cantora.''')

st.markdown('---')
st.markdown('# Sobre a Base de Dados')
st.markdown('''A obtenção das letras das músicas foi dada por scrapping do site [Genius](https://genius.com/). Após isso, foram passadas por um pré-processamento que pode ser visto [neste link](https://github.com/rianpf/DataScience/tree/main/streamlit). Com isso, as letras já estão em minúsculo, ausentes de pontuações e stopwords, além de já estarem lemmatizadas.

Com isso em mente, podemos olhar as informações que serão úteis para as análises aqui construídas:

__*Dicionário dos dados*__

- *album*: nome do álbum em que se insere a música;
- *title*: nome da música;
- *lyrics*: música tokenizada, já passada pelo pré-processamento;
- *lyrics_uppers*: música parcialmente passada pelo pré-processamento;
- *lyrics_str*: música não tokenizada, já passada pelo pré-processamento.
- *size*: tamanho da música''')
st.markdown('---')
st.markdown('### Base de dados:  Letras')

@st.cache(allow_output_mutation=True)
def load_metadata(url):
    df = pd.read_csv(url)
    df['lyrics'] = df['lyrics'].apply(eval)
    return df

df = load_metadata('lana_lyrics_clean.csv')
st.dataframe(df)

st.markdown('---')
st.title('Visualização Gráfica')
st.markdown('---')

html_temp_frequencia = """ <div style =background-image:url('https://rollingstone.uol.com.br/media/_versions/legacy/2015/img-1032751-lana-del-rey_widelg.jpg');padding:13px;border: 5px solid;"> 
    <h1 style ="color:#6efff6;font-family:helvetica;webkit-text-stroke: 2px black;margin-left:20px;text-align:left;">Frequência <br> de Palavras</h1> 
    </div> """
st.markdown(html_temp_frequencia, unsafe_allow_html = True) 

st.markdown(' ')
st.markdown('''A primeira análise que podemos fazer diz respeito às palavras mais frequentes na discografia da cantora:''')

lemmas = []

def lemma(df_series, lista):
 for sentence in df_series:
  for word in sentence:
    lista.append(word) 

lemma(df['lyrics'], lemmas)

frequency = nltk.FreqDist(lemmas)

@st.cache(hash_funcs={matplotlib.figure.Figure: hash})
def frequencia(frequency):
    fig, ax = plt.subplots() 
    ax = frequency.plot(30, title = "Frequência de Palavras", color = "magenta")
    return fig

st.pyplot(frequencia(frequency))
st.markdown('''Percebemos, inicialmente, que palavras como "like", "im" e "love" são frequentes em suas canções; no entanto, para quem está acostumado com as músicas da cantora, é de ciência que cada álbum possui uma temática diferente, o que talvez influencie o vocabulário.

Devido a isso, já já veremos individualmente como cada álbum se comporta em relação às palavras. :smile: ''')
lemmas_ldr = []
lemmas_ultra = []
lemmas_lfl = []
lemmas_cotcc = []
lemmas_btd = []
lemmas_hm = []
lemmas_nfr = []
lemmas_bb = []

lemma(df[df['album'] == 'Lana Del Ray']['lyrics'], lemmas_ldr)
lemma(df[df['album'] == 'Ultraviolence (Deluxe Edition)']['lyrics'], lemmas_ultra)
lemma(df[df['album'] == 'Lust for Life']['lyrics'], lemmas_lfl)
lemma(df[df['album'] == 'Chemtrails Over the Country Club']['lyrics'], lemmas_cotcc)
lemma(df[df['album'] == 'Born to Die - The Paradise Edition']['lyrics'], lemmas_btd)
lemma(df[df['album'] == 'Honeymoon']['lyrics'], lemmas_hm)
lemma(df[df['album'] == 'Norman Fucking Rockwell!']['lyrics'], lemmas_nfr)
lemma(df[df['album'] == 'Blue Banisters']['lyrics'], lemmas_bb)

frequency_ldr = nltk.FreqDist(lemmas_ldr)
frequency_ultra = nltk.FreqDist(lemmas_ultra)
frequency_lfl = nltk.FreqDist(lemmas_lfl)
frequency_cotcc = nltk.FreqDist(lemmas_cotcc)
frequency_btd = nltk.FreqDist(lemmas_btd)
frequency_hm = nltk.FreqDist(lemmas_hm)
frequency_nfr = nltk.FreqDist(lemmas_nfr)
frequency_bb = nltk.FreqDist(lemmas_bb)

col11, col12 = st.columns(2)
with col11:
    frequencia(frequency_ldr)
    frequencia(frequency_ultra)
    frequencia(frequency_lfl)
    frequencia(frequency_cotcc)
with col12:
    frequencia(frequency_btd)
    frequencia(frequency_hm)
    frequencia(frequency_nfr)
    frequencia(frequency_bb)

st.markdown('---')
html_temp_bigramas = """ <div style =background-image:url('https://www.billboard.com/wp-content/uploads/media/lana-del-rey-ultraviolence-2014-billboard-650.jpg?w=650');padding:13px;border: 5px solid;"> 
    <h1 style ="color:#d1d1d1;font-family:helvetica;webkit-text-stroke: 2px black;margin-left:60px;text-align:left;"> Bigramas </h1> 
    </div> """
st.markdown(html_temp_bigramas, unsafe_allow_html = True) 

st.markdown(' ')
st.markdown('''A próxima questão que podemos ver diz respeito aos bigramas presentes nas músicas, ou seja, os conjuntinhos de duas palavras que aparecem juntas.

Podemos, primeiramente, ver um gráfico que exibe os bigramas mais frequentes em toda a discografia da cantora:''')
bigramas = []
def ngrams(lista, ngram, size, df_series):
  for i in df.index:
    if len(df_series[i]) > size:
      lista.append([item for item in ngram(df_series[i])])
    else:
      lista.append([])
ngrams(bigramas, bigrams, 2, df['lyrics'])
bigrams_list = []
for sentence in bigramas:
  for bigrama in sentence:
    bigrams_list.append(bigrama)

bigrams_series = (pd.Series(bigrams_list).value_counts())

@st.cache(hash_funcs={matplotlib.figure.Figure: hash})
def bigramas_grafico1(bigrams_series):
    fig, ax = plt.subplots() 
    ax = bigrams_series[:30].plot.bar(figsize=(10,8), color = 'magenta')
    plt.title('30 Bigramas mais frequentes', fontsize = 16)
    return fig

st.pyplot(bigramas_grafico1(bigrams_series))

st.markdown('''Percebemos que algumas palavras são separadas devido ao pré-processamento, como "wanna" e "gonna"; além disso, os trechos utilizados apenas para a musicalidade, como "ah ah" e "oh oh", são bem frequentes.

No que diz respeito aos conjuntos de duas palavras realmente relevantes, vemos a existência de "god know", "feel like" e "im love", o que, de fato, reflete grande parte das composições da cantora, haja vista que recorrentemente refere-se à figura divina para explicitar suas intenções, além de possuir uma extrema subjetividade e grande relação de suas músicas ao amor.''')

st.markdown('---')
html_temp_wordcloud = """ <div style =background-image:url('https://lastfm.freetls.fastly.net/i/u/770x0/b64faaf0d7254a0297fb75e2bda3e9f5.jpg');padding:13px;border: 5px solid;"> 
    <h1 style ="color:#f0161d;font-family:helvetica;webkit-text-stroke: 2px black;margin-left:90px;text-align:left;">WordClouds</h1> 
    </div> """
st.markdown(html_temp_wordcloud, unsafe_allow_html = True) 

st.markdown(' ')
st.markdown('''O próximo ponto que trabalharemos reflete um ponto trabalhado acima: o fato de cada álbum possuir uma essência única, o que também é explicitado pelas diferentes composições.

Para vermos essa diferença, podemos avaliar a nuvem de palavras de cada álbum, para compará-los:''')

def WordNuvem(df_series): 
    '''
    Concatena todos os textos do DataSeries e produz um WordCloud dos resultados
    '''
    text = df_series.sum()
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color='black', width = 500, height = 300).generate(text)
    fig, ax = plt.subplots()
    plt.figure(figsize=(12,10), facecolor='k' )
    ax = plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

if st.button('Lana Del Ray  '):
    st.pyplot(WordNuvem(df[df['album'] == 'Lana Del Ray']['lyrics_str']))
if st.button('Born to Die - Paradise Edition'):
    st.pyplot(WordNuvem(df[df['album'] == 'Born to Die - The Paradise Edition']['lyrics_str']))
if st.button('Ultraviolence  '):
    st.pyplot(WordNuvem(df[df['album'] == 'Ultraviolence (Deluxe Edition)']['lyrics_str']))
if st.button('Honeymoon  '):
    st.pyplot(WordNuvem(df[df['album'] == 'Honeymoon']['lyrics_str'])) 
if st.button('Lust for Life  '):
    st.pyplot(WordNuvem(df[df['album'] == 'Lust for Life']['lyrics_str']))
if st.button('Norman Fucking Rockwell'):
    st.pyplot(WordNuvem(df[df['album'] == 'Norman Fucking Rockwell!']['lyrics_str']))  
if st.button('Chemtrails Over the Country Club  '):
    st.pyplot(WordNuvem(df[df['album'] == 'Chemtrails Over the Country Club']['lyrics_str']))
if st.button('Blue Banisters  '):
    st.pyplot(WordNuvem(df[df['album'] == 'Blue Banisters']['lyrics_str']))

st.markdown('---')   

html_temp_lexico = """ <div style =background-image:url('https://i0.wp.com/tracklist.com.br/wp-content/uploads/2017/07/lana-del-rey.jpg?fit=700%2C350&ssl=1');padding:13px;border: 5px solid;"> 
    <h1 style ="color:#c354ff;font-family:helvetica;webkit-text-stroke: 2px black;margin-left:40px;text-align:left;">Diversidade <br> Lexical</h1> 
    </div> """
st.markdown(html_temp_lexico, unsafe_allow_html = True) 

st.markdown(' ')
st.markdown('''Outra questão que podemos avaliar diz respeito ao outro caráter de Lana: compositora. O quão será que as suas composições são diversas lexicalmente, isto é, possuem palavras diferentes? Podemos fazer essa análise comparando a diversidade lexical de cada álbum e ver se, ao longo dos anos, as composições foram ganhando caráter menos genérico:''')

def lexical_diversity(texto):
  return len(set(texto)) / len(texto)

lexico_ldr = lexical_diversity(' '.join(df[df['album'] == 'Lana Del Ray']['lyrics_str']).split())
lexico_btd = lexical_diversity(' '.join(df[df['album'] == 'Born to Die - The Paradise Edition']['lyrics_str']).split())
lexico_ultra = lexical_diversity(' '.join(df[df['album'] == 'Ultraviolence (Deluxe Edition)']['lyrics_str']).split())
lexico_hm = lexical_diversity(' '.join(df[df['album'] == 'Honeymoon']['lyrics_str']).split())
lexico_lfl = lexical_diversity(' '.join(df[df['album'] == 'Lust for Life']['lyrics_str']).split())
lexico_nfk = lexical_diversity(' '.join(df[df['album'] == 'Norman Fucking Rockwell!']['lyrics_str']).split())
lexico_cotcc = lexical_diversity(' '.join(df[df['album'] == 'Chemtrails Over the Country Club']['lyrics_str']).split())
lexico_bb = lexical_diversity(' '.join(df[df['album'] == 'Blue Banisters']['lyrics_str']).split())

bmap = brewer2mpl.get_map('Set2','qualitative',8,reverse=True)
colors = bmap.mpl_colors

@st.cache(hash_funcs={matplotlib.figure.Figure: hash})
def diversidade_lexical():
    fig = plt.figure(figsize = (18,14))
    plt.bar(['Lana del Ray', 'Born to Die', 'Ultraviolence', 'Honeymoon', 'Lust for Life', 'Norman Fucking Rockwell', 'Chemtrails', 'Blue Banisters'], 
    [lexico_ldr*100, lexico_btd*100, lexico_ultra*100, lexico_hm*100, lexico_lfl*100, lexico_nfk*100, lexico_cotcc*100, lexico_bb*100], color = colors)
    plt.title('Diversidade lexical', fontsize = 16)
    plt.ylabel('Porcentagem (em %)')
    return fig

st.pyplot(diversidade_lexical())

st.markdown('''Como podemos ver no gráfico, em linhas gerais, as composições de Lana del Rey aumentaram sua diversidade lexical ao longo dos anos, já que a cantora inseriu em suas obras novas temáticas e, consequentemente, novo vocabulário. Tal ocasião só difere um pouco para o "Lana del Ray", mas é explicável, já que, nessa época, a cantora se apresentava até com outro nome artístico (possuía uma estética TOTALMENTE diferente), e também para o Lust for Life... mas a gente dá um desconto, já que é o primeiro álbum um pouquinho feliz da coitada rs''')

st.markdown('---')
html_temp_tamanho = """ <div style =background-image:url('https://i.pinimg.com/originals/ce/cc/9c/cecc9c64ac849b33c102c7581521d1e9.png');padding:13px;border: 5px solid;"> 
    <h1 style ="color:#e03f00;font-family:helvetica;webkit-text-stroke: 2px black;margin-left:40px;text-align:left;">Tamanho <br> do Texto</h1> 
    </div> """
st.markdown(html_temp_tamanho, unsafe_allow_html = True) 
st.markdown(' ')
st.markdown('''O próximo ponto de análise que traremos diz respeito ao tamanho das composições (em número de caracteres alfanuméricos).  

Para isso, primeiramente, podemos ver como cada música se comporta individualmente dentro de cada álbum:''')

@st.cache(hash_funcs={matplotlib.figure.Figure: hash})
def tamanho_musicas(df):
    fig = px.bar(df, y="album", x="size", color = "title", title="Tamanho das músicas por álbum")
    fig.update_layout(showlegend=False)
    fig.update_layout(yaxis_categoryarray = ["Blue Banisters","Chemtrails Over the Country Club", "Norman Fucking Rockwell!", "Lust for Life", "Honeymoon", 
    "Ultraviolence (Deluxe Edition)", "Born to Die - The Paradise Edition", "Lana Del Ray"])
    return fig

st.plotly_chart(tamanho_musicas(df))

st.markdown('''Percebemos, então, que o álbum mais extenso é o Born to Die, o que é justificável, já que, aqui, trabalhamos com o Paradise Edition, que contém músicas adicionais em relação ao normal.

Percebemos, também, que, em linhas gerais, o Lust for Life também apresenta grande extensão das composições, o que também é justificável pelo álbum apresentar mais músicas e, junto a isso, colaborações com outros artistas, com os rappers A$AP Rocky e Playboi Carti.

Voltando ao Born to Die, percebemos que, em relação às composições, apresenta grande parte das músicas mais extensas, como "Off to the Races", "Diet Mountain Dew" e "National Anthem". Quando se conhece um pouco da discografia da cantora, pode-se perceber que não necessariamente essas são as músicas mais longas. Por exemplo, no Ultraviolence, vemos "Cruel World", uma música com mais de 6 minutos, ou no Norman Fucking Rockwell, vemos "Venice Bitch", com quase 10 minutos. Isso reflete o fato de que as maiores músicas, em tempo, da cantora apresentam muitos elementos musicais, mas que não necessariamente são refletidos nas composições.

Podemos, em seguida, ver o tamanho médio de uma música em cada álbum, para entendermos essa análise em um panorama mais geral para cada álbum:''')

@st.cache(hash_funcs={matplotlib.figure.Figure: hash})
def tamanho_poralbum():
    fig = px.bar(df.groupby(['album']).mean().reset_index(), y="album", x="size",  title="Tamanho médio das músicas por álbum", color = "album")
    fig.update_layout(showlegend=False)
    fig.update_layout(yaxis_categoryarray = ["Blue Banisters","Chemtrails Over the Country Club", "Norman Fucking Rockwell!", "Lust for Life", "Honeymoon", 
    "Ultraviolence (Deluxe Edition)", "Born to Die - The Paradise Edition", "Lana Del Ray"])
    return fig

st.plotly_chart(tamanho_poralbum())

st.markdown('''É possível ver, no gráfico acima, que o álbum com maior tamanho médio de música é o Lust for Life, seguido pelo Born to Die.

Apesar de o Born to Die apresentar muitas músicas com grande extensão, também apresenta várias com uma extensão pequena, o que faz com que o Lust for Life tenha um tamanho médio maior, haja vista que não possui tantas composições curtas.''')

st.markdown('---')
html_temp_predicao = """ <div style =background-image:url('https://static.stereogum.com/blogs.dir/2/files/2011/11/Lana-Del-Rey.jpg');padding:13px;border: 5px solid;"> 
    <h1 style ="color:#ff304f;font-family:helvetica;webkit-text-stroke: 2px black;margin-right:40px;text-align:right;">Predição</h1> 
    </div> """
st.markdown(html_temp_predicao, unsafe_allow_html = True) 

st.markdown(' ')
st.markdown('''Após a análise das WordClouds e, com um conhecimento prévio a respeito da discografia de Lana Del Rey, conseguimos claramente separar a carreira da cantora em dois momentos: antes e após de Lust for Life.

Antes do lançamento deste álbum, as letras e melodias da cantora apresentavam-se de forma mais melancólica, o que, em contraponto, após o lançamento do álbum, começou a ser trocado por novas estéticas não tão tristes (não estou dizendo que são felizes, só são mais animadinhas rsrsrs), o que pode ser exemplificado até pelo álbum de Country recente da cantora.

Dito isso, trabalharemos com um modelo que prediz, dado um trecho de uma música, se ela se encaixaria em um momento anterior ou posterior a esse álbum.

Você pode digitar o que quiser no campo a seguir :smile: :''')

count_vectorizer = CountVectorizer(binary=True)
X_BOW = count_vectorizer.fit_transform(df['lyrics_str'])

df_cv = pd.DataFrame(X_BOW.toarray(), columns = count_vectorizer.get_feature_names())

df['binary']= np.nan

for i in df.index:
    if df['album'][i] == 'Lana Del Ray':
        df['binary'][i] = 0
    elif df['album'][i] == 'Born to Die - The Paradise Edition':
        df['binary'][i] = 0
    elif df['album'][i] == 'Ultraviolence (Deluxe Edition)':
        df['binary'][i] = 0
    elif df['album'][i] == 'Honeymoon':
        df['binary'][i] = 0
    else:
        df['binary'][i] = 1

X_bow = X_BOW.toarray()
y_bow = df['binary']

X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(X_bow, y_bow, test_size = 0.2)

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_bow, y_train_bow)
naive_bayes_pred_bow = naive_bayes.predict(X_test_bow)

new_lyrics = st.text_input("Escreva o trecho da musica:", "Digite aqui ...")

teste = count_vectorizer.transform([new_lyrics])

if(st.button('Predizer')):
    result = naive_bayes.predict(teste)
    if result == 0:
        st.error('Seria uma vibe meio triste, anterior ao Lust for Life :cry:')
        st.markdown("![Alt Text](https://c.tenor.com/W4n2MhGUnP8AAAAC/lana-del-rey-sad.gif)")
    elif result == 1:
        st.success('Seria algo mais felizinho, posterior ao Lust for Life :smile:')
        st.markdown("![Alt Text](https://media2.giphy.com/media/TCXZYGKKyRjz2/giphy.gif)")

st.markdown('---')
html_temp_sent = """ <div style =background-image:url('https://asset.kompas.com/crops/n9HpNFkWQyYhUY-rkds19BwHCyM=/0x0:999x666/780x390/filters:watermark(data/photo/2020/03/10/5e6775d943eeb.png,0,-0,1)/data/photo/2018/02/04/189030224.jpg');padding:13px;border: 5px solid;"> 
    <h1 style ="color:#fcdfb3;font-family:helvetica;webkit-text-stroke: 2px black;margin-right:30px;text-align:right;">Análise de <br> Sentimento</h1> 
    </div> """
st.markdown(html_temp_sent, unsafe_allow_html = True) 

st.markdown(' ')
st.markdown(''' Outra análise que podemos fazer diz respeito ao sentimento existente em cada canção.

Para isso, utilizaremos a biblioteca "TextBlob", que nos retornará dois valores:

- *polarity*: valor que varia de -1 a 1 e reflete a polaridade do texto, sendo que, quanto mais próximo de -1, mais negativo ele é e, consequentemente, quanto mais próximo de 1, mais positivo ele é;
- *subjectivity*: valor que varia de 0 a 1 e reflete a subjetividade do texto, sendo que, quanto mais próximo de 0, mais objetivo é o texto, isto é, possui menos opiniões pessoais e, consequentemente, quanto mais próximo de 1, mais subjetivo ele é, isto é, possui mais opiniões pessoais.

Para ver o sentimento de cada composição, você deve, primeiramente, selecionar o álbum desejado para, posteriormente, selecionar a canção :blush: :''')

if st.checkbox('Lana Del Ray '):
    for i in df[df['album'] == 'Lana Del Ray']['title'].unique():
        if st.button(i):
            for j in df[df['title'] == i].index:
                answer = TextBlob(df[df['title'] == i]['lyrics_str'][j]).sentiment
                st.write(answer)
if st.checkbox('Born to Die - Paradise Edition '):
    for i in df[df['album'] == 'Born to Die - The Paradise Edition']['title'].unique():
        if st.button(i):
            for j in df[df['title'] == i].index:
                answer = TextBlob(df[df['title'] == i]['lyrics_str'][j]).sentiment
                st.write(answer)
if st.checkbox('Ultraviolence '):
    for i in df[df['album'] == 'Ultraviolence (Deluxe Edition)']['title'].unique():
        if st.button(i):
            for j in df[df['title'] == i].index:
                answer = TextBlob(df[df['title'] == i]['lyrics_str'][j]).sentiment
                st.write(answer)
if st.checkbox('Honeymoon '):
    for i in df[df['album'] == 'Honeymoon']['title'].unique():
        if st.button(i):
            for j in df[df['title'] == i].index:
                answer = TextBlob(df[df['title'] == i]['lyrics_str'][j]).sentiment
                st.write(answer)
if st.checkbox('Lust for Life '):
    for i in df[df['album'] == 'Lust for Life']['title'].unique():
        if st.button(i):
            for j in df[df['title'] == i].index:
                answer = TextBlob(df[df['title'] == i]['lyrics_str'][j]).sentiment
                st.write(answer)
if st.checkbox('Norman Fucking Rockwell '):
    for i in df[df['album'] == 'Norman Fucking Rockwell!']['title'].unique():
        if st.button(i):
            for j in df[df['title'] == i].index:
                answer = TextBlob(df[df['title'] == i]['lyrics_str'][j]).sentiment
                st.write(answer)
if st.checkbox('Chemtrails Over the Country Club '):
    for i in df[df['album'] == 'Chemtrails Over the Country Club']['title'].unique():
        if st.button(i):
            for j in df[df['title'] == i].index:
                answer = TextBlob(df[df['title'] == i]['lyrics_str'][j]).sentiment
                st.write(answer)
if st.checkbox('Blue Banisters '):
    for i in df[df['album'] == 'Blue Banisters']['title'].unique():
        if st.button(i):
            for j in df[df['title'] == i].index:
                answer = TextBlob(df[df['title'] == i]['lyrics_str'][j]).sentiment
                st.write(answer)

st.markdown('---')
html_temp_ent = """ <div style =background-image:url('https://i.pinimg.com/originals/ce/e3/67/cee36765ae1df42be13e218625190b60.png');padding:13px;border: 5px solid;"> 
    <h1 style ="color:#59d455;font-family:helvetica;webkit-text-stroke: 2px black;margin-left:2px;text-align:left;">Entidades <br> Extraídas do Texto</h1> 
    </div> """
st.markdown(html_temp_ent, unsafe_allow_html = True) 

st.markdown(' ')
st.markdown('''Por fim, podemos analisar as Entidades Extraídas do Texto.

Para isso, utilizaremos, aqui, a biblioteca "Spacy".

Para selecionar a canção, você deve, primeiramente, selecionar o álbum :wink: :''')

nlp = spacy.load('en_core_web_sm')

if st.checkbox(' Lana Del Ray '):
    for i in df[df['album'] == 'Lana Del Ray']['title'].unique():
        if st.button(f'{i} '):
            for j in df[df['title'] == i].index:
                doc = nlp(df[df['title'] == i]['lyrics_str'][j])
                ent_html = displacy.render(doc, style='ent', jupyter=False)
                st.markdown(ent_html, unsafe_allow_html=True)
if st.checkbox(' Born to Die - Paradise Edition '):
    for i in df[df['album'] == 'Born to Die - The Paradise Edition']['title'].unique():
        if st.button(f'{i} '):
            for j in df[df['title'] == i].index:
                doc = nlp(df[df['title'] == i]['lyrics_str'][j])
                ent_html = displacy.render(doc, style='ent', jupyter=False)
                st.markdown(ent_html, unsafe_allow_html=True)
if st.checkbox(' Ultraviolence '):
    for i in df[df['album'] == 'Ultraviolence (Deluxe Edition)']['title'].unique():
        if st.button(f'{i} '):
            for j in df[df['title'] == i].index:
                doc = nlp(df[df['title'] == i]['lyrics_str'][j])
                ent_html = displacy.render(doc, style='ent', jupyter=False)
                st.markdown(ent_html, unsafe_allow_html=True)
if st.checkbox(' Honeymoon '):
    for i in df[df['album'] == 'Honeymoon']['title'].unique():
        if st.button(f'{i} '):
            for j in df[df['title'] == i].index:
                doc = nlp(df[df['title'] == i]['lyrics_str'][j])
                ent_html = displacy.render(doc, style='ent', jupyter=False)
                st.markdown(ent_html, unsafe_allow_html=True)
if st.checkbox(' Lust for Life '):
    for i in df[df['album'] == 'Lust for Life']['title'].unique():
        if st.button(f'{i} '):
            for j in df[df['title'] == i].index:
                doc = nlp(df[df['title'] == i]['lyrics_str'][j])
                ent_html = displacy.render(doc, style='ent', jupyter=False)
                st.markdown(ent_html, unsafe_allow_html=True)
if st.checkbox(' Norman Fucking Rockwell '):
    for i in df[df['album'] == 'Norman Fucking Rockwell!']['title'].unique():
        if st.button(f'{i} '):
            for j in df[df['title'] == i].index:
                doc = nlp(df[df['title'] == i]['lyrics_str'][j])
                ent_html = displacy.render(doc, style='ent', jupyter=False)
                st.markdown(ent_html, unsafe_allow_html=True)
if st.checkbox(' Chemtrails Over the Country Club '):
    for i in df[df['album'] == 'Chemtrails Over the Country Club']['title'].unique():
        if st.button(f'{i} '):
            for j in df[df['title'] == i].index:
                doc = nlp(df[df['title'] == i]['lyrics_str'][j])
                ent_html = displacy.render(doc, style='ent', jupyter=False)
                st.markdown(ent_html, unsafe_allow_html=True)
if st.checkbox(' Blue Banisters '):
    for i in df[df['album'] == 'Blue Banisters']['title'].unique():
        if st.button(f'{i} '):
            for j in df[df['title'] == i].index:
                doc = nlp(df[df['title'] == i]['lyrics_str'][j])
                ent_html = displacy.render(doc, style='ent', jupyter=False)
                st.markdown(ent_html, unsafe_allow_html=True)

st.markdown('---')
st.markdown('# Conclusão')
st.markdown('''Bom, como vimos rapidamente por essas análises, ao longo dos anos, não só a estética musical de Lana del Rey foi alterada, mas também a forma como suas composições ocorreram. 

Com isso e o conhecimento sobre a história de cada álbum, podemos tirar conclusões interessantes a respeito de cada era da cantora utilizando ferramentas de Machine Learning. 

Espero que você tenha conseguido acompanhar passo a passo. Para acessar os códigos, o link para o repositório no GitHub está na barra lateral à esquerda.

Obrigado por chegar até aqui! :heart: ''')

st.markdown("![Alt Text](https://media2.giphy.com/media/l4FGnkt4OpJf5JcJy/giphy.gif?cid=6c09b9529f035c6822fc532fe1613849f713daab568d785d&rid=giphy.gif&ct=g)")

image = Image.open('lana_turing.jpg')
st.sidebar.image(image,use_column_width=True)
st.sidebar.markdown('''Projeto feito para a área de Data Science para o Turing USP.

Feito por : Rian Fernandes''')
st.sidebar.markdown("- [Github](https://github.com/rianpf)")