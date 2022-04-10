# import required packages

# data manipulation
from fastapi.encoders import jsonable_encoder
import re
import numpy as np
import pandas as pd

# ignore chain assignment warning 
pd.options.mode.chained_assignment = None


# lyrics retrieval
genius_token = 'JhNas_g4xHymyA1mzCmZ_alk-MpiOVrfihxvlQ4NYjOe2e_XpBRZB7dCgHG3PSij'
import lyricsgenius

# nltk- text tokenization
import nltk
from nltk.tokenize import sent_tokenize

# nltk- vader
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy

# nrc- emotion lexicon scores
from nrclex import NRCLex

# bert- transformer pipelines
from transformers import pipeline

# spotify
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# database
from sqlalchemy import create_engine
import psycopg2 
import io

def pydantic_df(model):
    jdata = jsonable_encoder(model)
    genius = lyricsgenius.Genius(genius_token)
    sdf = pd.DataFrame()
    # raw lyrics data
    lyrics_raw = []
    # song titles
    titles = []
    # song artists
    artists = []
    # isrc numbers
    isrcs = []

    for i in range(len(jdata['data'])):
        n = jdata['data'][i]['attributes']['name']
        #shorten search query in genius due to limitation 
        n_short = re.sub(r'\(.*', '', n)
        a = jdata['data'][i]['attributes']['artistName']
        isrc = jdata['data'][i]['attributes']['isrc']
        try:
            song = genius.search_song(n_short, a)
            ly = song.lyrics
        except:
            ly = None
        lyrics_raw.append(ly)
        titles.append(n)
        artists.append(a)
        isrcs.append(isrc)
    sdf['isrc'] = isrcs
    sdf['title'] = titles
    sdf['artist'] = artists
    sdf['lyrics'] = lyrics_raw

    # filter out the songs that have no lyrics
    sdf = sdf.dropna()
    return sdf

def lyrics_clean(songs_df):
    # cleaned lyrics, each song's lyrics as a list of strings
    lyrics_lines = []
    # cleaned lyrics, each song's lyrics as one string
    lyrics_string = []
    for lyrics in songs_df['lyrics']:
        lyrics = re.sub(r'\n+', '\n', lyrics)
        lyrics = lyrics.replace('\n', '. ')
        lyrics = lyrics.replace('\u2005', '')
        lyrics = re.sub(r'\[[^\]]*\][.]', '', lyrics)
        # lyrics as one string
        ly_concat = lyrics
        lyrics_string.append(ly_concat)
        # return lines as list of strings
        ly_list = sent_tokenize(lyrics)
        lyrics_lines.append(ly_list)

    songs_df['lyrics_lines'] = lyrics_lines
    songs_df['lyrics_string'] = lyrics_string
    return songs_df