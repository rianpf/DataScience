import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import brewer2mpl
import nltk
from nltk.text import Text
from nltk import bigrams
from nltk import trigrams
from PIL import Image

st.title('Análise das músicas da Lana Del Rey')

df = pd.read_csv('lana_lyrics_clean.csv')

st.markdown('---')
st.markdown('# Sobre a Cantora')

foto = Image.open('lanadelrey.jpg')

st.image(foto, caption="Lana na capa do seu mais recente álbum, 'Blue Banisters'", width = 690)

st.markdown('''Lana Del Rey é uma cantora norte-americana de Indie Pop, que carrega influências do Rock, Hip Hop, Jazz, Blues e, mais recentemente, até Country. Seu primeiro álbum digital, lançado em 2010, deu início à carreira que mantém até hoje, totalizando 8 albuns gravados.

Ao passar dos anos, a estética da cantora foi adquirindo diferentes formas, resultado de mudanças em sua vida pessoal e forma de enxergar o universo. Tais mudanças, portanto, são notáveis em cada álbum.

O intuito desta amostragem é, utilizando técnicas de Processamento de Linguagem Natural, Machine Learning e Data Science, explicitar algumas de tais alterações na arte da cantora.''')

st.markdown('---')
st.markdown('# Sobre a Base de Dados')
st.markdown('''A obtenção das letras das músicas foi dada por scrapping do site [Genius](https://genius.com/). Após isso, foram passadas por um pré-processamento que pode ser visto [neste link](). Com isso, as letras já estão em minúsculo, ausentes de pontuações e stopwords, além de já estarem lemmatizadas.

Com isso em mente, podemos olhar as informações que serão úteis para as análises aqui construídas:

__*Dicionário dos dados*__

- *album*: nome do álbum em que se insere a música;
- *title*: nome da música;
- *lyrics*: música tokenizada, já passada pelo pré-processamento;
- *lyrics_str*: música não tokenizada, já passada pelo pré-processamento.''')

st.markdown('---')
st.markdown('### Base de dados:  Letras')
st.dataframe(df)

st.markdown('---')
st.title('Visualização Gráfica')

st.markdown('''A primeira análise que podemos fazer diz respeito às palavras mais frequentes na discografia da cantora:''')

df['lyrics'] = df['lyrics'].apply(eval)

lemmas = []

def lemma(df_series, lista):
 for sentence in df_series:
  for word in sentence:
    lista.append(word) 

lemma(df['lyrics'], lemmas)

frequency = nltk.FreqDist(lemmas)

fig, ax = plt.subplots() 
ax = frequency.plot(30, title = "Frequência de Palavras", color = "purple")
st.pyplot(fig)
st.markdown('''Percebemos, inicialmente, que palavras como "like", "im" e "love" são frequentes em suas canções; no entanto, para quem está acostumado com as músicas da cantora, é de ciência que cada álbum possui uma temática diferente, o que talvez influencie o vocabulário.

Portanto, podemos ver, individualmente, as frequências de palavras para cada álbum:''')
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
    fig, ax = plt.subplots() 
    ax = frequency_ldr.plot(30, title = "Frequência de Palavras no 'Lana del Ray'", color = "purple")
    st.pyplot(fig)

    fig, ax = plt.subplots() 
    ax = frequency_ultra.plot(30, title = "Frequência de Palavras no 'Ultraviolence'", color = "purple")
    st.pyplot(fig)

    fig, ax = plt.subplots() 
    ax = frequency_lfl.plot(30, title = "Frequência de Palavras no 'Lust for Life'", color = "purple")
    st.pyplot(fig)

    fig, ax = plt.subplots() 
    ax = frequency_cotcc.plot(30, title = "Frequência de Palavras no 'Chemtrails Over the Country Club'", color = "purple")
    st.pyplot(fig)
with col12:
    fig, ax = plt.subplots() 
    ax = frequency_btd.plot(30, title = "Frequência de Palavras no 'Born to Die'", color = "purple")
    st.pyplot(fig)

    fig, ax = plt.subplots() 
    ax = frequency_hm.plot(30, title = "Frequência de Palavras no 'Honeymoon'", color = "purple")
    st.pyplot(fig)

    fig, ax = plt.subplots() 
    ax = frequency_nfr.plot(30, title = "Frequência de Palavras no 'Norman Fucking Rockwell'", color = "purple")
    st.pyplot(fig)

    fig, ax = plt.subplots() 
    ax = frequency_bb.plot(30, title = "Frequência de Palavras no 'Blue Banisters'", color = "purple")
    st.pyplot(fig)

st.markdown('''Percebemos que as palavras que já eram frequentes em todas as músicas de Lana continuam, em vias gerais, frequentes em cada álbum.

Mesmo assim, cada álbum possui suas especificidades: em "Lana del Ray", por exemplo, vemos "jump" e "gramma"; por outro lado, em "Ultraviolence" percebemos algumas palavras que remetem à estética não tão feliz da era, como "sad", "gun" e "yayo".

Em "Lust for Life", por outro lado, vemos algumas palavras que remetem a uma felicidade, diferente do outro álbum citado, como "alive" e "summer"; também vale ressaltar que este é o álbum que cita mais a palavra "love".

Percebemos a tendência que as músicas de Lana del Rey adquirem em um cenário pós-Honeymoon, em que as letras não tratam apenas de um contexto triste, mas vendo pelo menos algum sinal de positividade em cada composição.

Podemos, de uma forma mais didática, ver a importância de cada palavra, no que diz respeito às suas ocorrências quantitativamente, em uma WordCloud:''')

def WordNuvem(df_series, album): 
    '''
    Concatena todos os textos do DataSeries e produz um WordCloud dos resultados
    '''
    text = df_series.sum()
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color='black', width = 500, height = 300).generate(text)
    fig, ax = plt.subplots()
    plt.figure(figsize=(12,10))
    ax = plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f'Wordcloud do{album}', fontsize = 22)

st.pyplot(WordNuvem(df['lyrics_str'],"s álbuns"))

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown('''Também podemos ver tais ocorrências separadamente por álbum, o que corrobora os gráficos de frequência já mostrados acima:''')

col21, col22 = st.columns(2)
with col21 :
    st.pyplot(WordNuvem(df[df['album'] == 'Lana Del Ray']['lyrics_str']," 'Lana Del Ray'"))
    st.pyplot(WordNuvem(df[df['album'] == 'Ultraviolence (Deluxe Edition)']['lyrics_str']," 'Ultraviolence'"))
    st.pyplot(WordNuvem(df[df['album'] == 'Lust for Life']['lyrics_str']," 'Lust for Life'"))
    st.pyplot(WordNuvem(df[df['album'] == 'Chemtrails Over the Country Club']['lyrics_str']," 'Chemtrails Over the Country Club'"))

with col22 :   
    st.pyplot(WordNuvem(df[df['album'] == 'Born to Die - The Paradise Edition']['lyrics_str']," 'Born to Die'"))
    st.pyplot(WordNuvem(df[df['album'] == 'Honeymoon']['lyrics_str']," 'Honeymoon'")) 
    st.pyplot(WordNuvem(df[df['album'] == 'Norman Fucking Rockwell!']['lyrics_str']," 'Norman Fucking Rockwell'"))  
    st.pyplot(WordNuvem(df[df['album'] == 'Blue Banisters']['lyrics_str']," 'Blue Banisters'"))

st.markdown('''Por fim, outra questão que podemos avaliar diz respeito ao outro caráter de Lana: compositora. O quão será que as suas composições são diversas lexicalmente, isto é, possuem palavras diferentes? Podemos fazer essa análise comparando a diversidade lexical de cada álbum e ver se, ao longo dos anos, as composições foram ganhando caráter menos genérico:''')

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

fig = plt.figure(figsize = (18,12))
plt.bar(['Lana del Ray', 'Born to Die', 'Ultraviolence', 'Honeymoon', 'Lust for Life', 'Norman Fucking Rockwell', 'Chemtrails', 'Blue Banisters'], 
[lexico_ldr*100, lexico_btd*100, lexico_ultra*100, lexico_hm*100, lexico_lfl*100, lexico_nfk*100, lexico_cotcc*100, lexico_bb*100], color = colors)
plt.title('Diversidade lexical', fontsize = 16)
plt.ylabel('Porcentagem (em %)')
st.pyplot(fig)

st.markdown('''Como podemos ver no gráfico, em linhas gerais, as composições de Lana del Rey aumentaram sua diversidade lexical ao longo dos anos, já que a cantora inseriu em suas obras novas temáticas e, consequentemente, novo vocabulário. Tal ocasião só difere um pouco para o "Lana del Ray", mas é explicável, já que, nessa época, a cantora se apresentava até com outro nome artístico (possuía uma estética TOTALMENTE diferente), e também para o Lust for Life... mas a gente dá um desconto, já que é o primeiro álbum um pouquinho feliz da coitada rs''')

st.markdown('---')
st.markdown('# Conclusão')
st.markdown('''Bom, como vimos rapidamente por essas análises, ao longo dos anos, não só a estética musical de Lana del Rey foi alterada, mas também a forma como suas composições ocorreram. 

Com isso e o conhecimento sobre a história de cada álbum, podemos tirar conclusões interessantes a respeito de cada era da cantora. 

Dito isso, podemos, em algum dia, fazer essa correspondência mais incisivamente :) Obrigado por chegar até aqui!''')

st.markdown("![Alt Text](https://media2.giphy.com/media/l4FGnkt4OpJf5JcJy/giphy.gif?cid=6c09b9529f035c6822fc532fe1613849f713daab568d785d&rid=giphy.gif&ct=g)")

image = Image.open('lana.jpg')
st.sidebar.image(image,use_column_width=True)

st.sidebar.markdown('Feito por : Rian Fernandes')
st.sidebar.markdown("- [Github](https://github.com/rianpf)")