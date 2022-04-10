# GET return information
# POST create something new
# PUT update
# DELETE DELETE 
# import required packages

# fastapi
from typing import List, Optional
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

# other scripts in the app
from songs_lyrics import pydantic_df, lyrics_clean
from songs_insights import nrc_scores, vader_scores, spot_scores, albert_scores

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

app = FastAPI()
sdf = pd.DataFrame()
ins_df = pd.DataFrame()
genius = lyricsgenius.Genius(genius_token)


class Preview(BaseModel):
    url: str


class Artwork(BaseModel):
    width: int
    height: int
    url: str
    bgColor: str
    textColor1: str
    textColor2: str
    textColor3: str
    textColor4: str


class PlayParams(BaseModel):
    id: str
    kind: str


class Attributes(BaseModel):
    previews: List[Preview]
    artwork: Artwork
    artistName: str
    url: str
    discNumber: int
    genreNames: List[str]
    durationInMillis: int
    releaseDate: str
    name: str
    isrc: str
    hasLyrics: bool
    albumName: str
    playParams: PlayParams
    trackNumber: int
    composerName: Optional[str] = None
    contentRating: Optional[str] = None


class Datum(BaseModel):
    id: str
    type: str
    href: str
    attributes: Attributes


class Model(BaseModel):
    next: str
    data: List[Datum]

class SongList(BaseModel):
    song_names: List[str]

dic = {
    'key1' : 'value1',
    'key2' : 'value2',
    'key3' : 'value3',
}

@app.get('/')
async def root():
    return {'message': 'Hello World!'}


@app.post('/music-insights')
async def post_req(model: Model):
    songs_df = pydantic_df(model)
    songs_df_clean = lyrics_clean(songs_df)
    sdf['nrc_score'] = nrc_scores(songs_df_clean)
    sdf['line_albert_score'] = albert_scores(songs_df_clean)
    sdf['str_vader_score'], sdf['line_vader_score']  = vader_scores(songs_df_clean)
    sdf['spot_audio_analysis'], sdf['spot_audio_features'] = spot_scores(songs_df_clean)
    
    
    ins_df['artist'] = songs_df_clean['artist']
    ins_df['title'] = songs_df_clean['title']

    ins_df['nrc_fear'] = [nrc_score[0] for nrc_score in sdf['nrc_score']]
    ins_df['nrc_anger'] = [nrc_score[1] for nrc_score in sdf['nrc_score']]
    ins_df['nrc_anticipation'] = [nrc_score[2] for nrc_score in sdf['nrc_score']]
    ins_df['nrc_trust'] = [nrc_score[3] for nrc_score in sdf['nrc_score']]
    ins_df['nrc_surprise'] = [nrc_score[4] for nrc_score in sdf['nrc_score']]
    ins_df['nrc_positive'] = [nrc_score[5] for nrc_score in sdf['nrc_score']]
    ins_df['nrc_negative'] = [nrc_score[6] for nrc_score in sdf['nrc_score']]
    ins_df['nrc_sadness'] = [nrc_score[7] for nrc_score in sdf['nrc_score']]
    ins_df['nrc_disgust'] = [nrc_score[8] for nrc_score in sdf['nrc_score']]
    ins_df['nrc_joy'] = [nrc_score[9] for nrc_score in sdf['nrc_score']]

    ins_df['line_vader_neg'] = [line_vader_score[0] for line_vader_score in sdf['line_vader_score']]
    ins_df['line_vader_neu'] = [line_vader_score[1] for line_vader_score in sdf['line_vader_score']]
    ins_df['line_vader_pos'] = [line_vader_score[2] for line_vader_score in sdf['line_vader_score']]
    ins_df['line_vader_compound'] = [line_vader_score[3] for line_vader_score in sdf['line_vader_score']]

    vader_str_df = sdf.str_vader_score.dropna().apply(pd.Series)
    ins_df['str_vader_neg'] = vader_str_df[vader_str_df.columns[0]]
    ins_df['str_vader_neu'] = vader_str_df[vader_str_df.columns[1]]
    ins_df['str_vader_pos'] = vader_str_df[vader_str_df.columns[2]]
    ins_df['str_vader_compound'] = vader_str_df[vader_str_df.columns[3]]

    spot_audio_ft_df = sdf.spot_audio_features.dropna().apply(pd.Series)
    ins_df['spot_danceability'] = spot_audio_ft_df[spot_audio_ft_df.columns[0]]
    ins_df['spot_energy'] = spot_audio_ft_df[spot_audio_ft_df.columns[1]]
    ins_df['spot_key'] = spot_audio_ft_df[spot_audio_ft_df.columns[2]]
    ins_df['spot_loudness'] = spot_audio_ft_df[spot_audio_ft_df.columns[3]]
    ins_df['spot_mode'] = spot_audio_ft_df[spot_audio_ft_df.columns[4]]
    ins_df['spot_speechiness'] = spot_audio_ft_df[spot_audio_ft_df.columns[5]]
    ins_df['spot_acousticness'] = spot_audio_ft_df[spot_audio_ft_df.columns[6]]
    ins_df['spot_instrumentalness'] = spot_audio_ft_df[spot_audio_ft_df.columns[7]]
    ins_df['spot_liveness'] = spot_audio_ft_df[spot_audio_ft_df.columns[8]]
    ins_df['spot_valence'] = spot_audio_ft_df[spot_audio_ft_df.columns[9]]
    ins_df['spot_tempo'] = spot_audio_ft_df[spot_audio_ft_df.columns[10]]

    ins_df['line_albert_sadness'] = [line_albert_score[0] for line_albert_score in sdf['line_albert_score']]
    ins_df['line_albert_joy'] = [line_albert_score[1] for line_albert_score in sdf['line_albert_score']]
    ins_df['line_albert_love'] = [line_albert_score[2] for line_albert_score in sdf['line_albert_score']]
    ins_df['line_albert_anger'] = [line_albert_score[3] for line_albert_score in sdf['line_albert_score']]
    ins_df['line_albert_fear'] = [line_albert_score[4] for line_albert_score in sdf['line_albert_score']]
    ins_df['line_albert_surprise'] = [line_albert_score[5] for line_albert_score in sdf['line_albert_score']]
    
    ins = ins_df.to_json(orient="split")
    return ins

