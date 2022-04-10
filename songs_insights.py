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

def nrc_scores(sdf):   
    print('start running nrc')
    # NRC emotional lexicon analysis
    nrc_scores = []
    for lines in sdf['lyrics_lines']:
        # nrc affection score list
        nrc_affection_freq_l = []
        for line in lines:
            nrc_txt = NRCLex(line)
            aff_dict = nrc_txt.affect_frequencies
    
            try: 
                aff_dict['anticip'] = aff_dict['anticipation']
            except: 
                pass
            
            aff_dict.pop('anticipation', None)

            affection_freq = list(aff_dict.values())
        

            # print(affection_freq)
            nrc_affection_freq_l.append(affection_freq)
            # affection_freq_arr = np.append(affection_freq_arr, np.array([affection_freq]), axis=0)

        ly_aff_arr = np.array(nrc_affection_freq_l)
        nrc_score = np.sum(ly_aff_arr, axis=0)
        nrc_scores.append(nrc_score)

    sdf['nrc_score'] = nrc_scores
    return sdf['nrc_score']

def vader_scores(sdf):
    print('start running vader')
    # Vader
    vader_sent = SentimentIntensityAnalyzer()
    # vader scores based on string
    sdf['str_vader_score'] = sdf['lyrics_string'].apply(lambda lyrics: vader_sent.polarity_scores(lyrics))

    # vader scores based on lines
    line_vader_scores = []
    for lines in sdf['lyrics_lines']:
        # nrc affection score list
        vader_score_l = []
        for line in lines:
            line_vader_score = vader_sent.polarity_scores(line)
            line_vader_score = list(line_vader_score.values())
            vader_score_l.append(line_vader_score)

        ly_vad_arr = np.array(vader_score_l)
        vader_score = np.sum(ly_vad_arr, axis=0)
        line_vader_scores.append(vader_score)

    sdf['line_vader_score'] = line_vader_scores
    return sdf['str_vader_score'], sdf['line_vader_score']

def spot_scores(sdf):
    # Spotify audio feature/ analysis
    # initiate spotipy- spotify api python wrapper
    print('start running spot')
    auth_manager = SpotifyClientCredentials()
    spot = spotipy.Spotify(auth_manager=auth_manager)
    spot_audio_analysis = []
    spot_audio_features = []
    for idx, song in sdf.iterrows():
        title = song['title']
        # in case
        t_short = re.sub(r'\(.*', '', title)
        isrc = song['isrc']
        artist = song['artist']
        ### REMINDER: ISRC SEARCH METHOD IS NOT STABLE AS SOME ISRC CODES PROVIDED BY APPLE MAY BE OUTDATED THUS COULD NOT BE USED ###
        # spot_track_obj = sp.search(q='isrc:' + isrc, type='track')
        try:
            spot_track_obj = spot.search(q='isrc:' + isrc, type='track')
            spot_track_id = spot_track_obj['tracks']['items'][0]['id']
            spot_ana = spot.audio_analysis(spot_track_id)
            spot_feat = spot.audio_features(tracks=[spot_track_id])
        except:
            try:
                spot_track_obj = spot.search(q='artist:' + artist + ' track:' + t_short, type='track')
                spot_track_id = spot_track_obj['tracks']['items'][0]['id']
                spot_ana = spot.audio_analysis(spot_track_id)
                spot_feat = spot.audio_features(tracks=[spot_track_id])
            except:
                spot_ana = None
                spot_feat = None
        

        spot_audio_analysis.append(spot_ana)
        spot_audio_features.append(spot_feat[0])

    sdf['spot_audio_analysis'] = spot_audio_analysis
    sdf['spot_audio_features'] = spot_audio_features
    return sdf['spot_audio_analysis'], sdf['spot_audio_features']

# BERT- ALBERT pretrained model
#TODO: albert scores based on string
#str_albert_scores = []
#for lyrics in sdf['lyrics_string']:
    #albert_pred_raw  = bert_cls(lyrics)
    #albert_score = [score_label['score'] for score_label in albert_pred_raw[0]]
    #str_albert_scores.append(albert_score)
#sdf['str_albert_score'] = str_albert_scores

# albert scores based on lines
def albert_scores(sdf):
    print('start running bert')
    bert_cls = pipeline("text-classification",model= 'bhadresh-savani/albert-base-v2-emotion' , return_all_scores=True)
    print('created classifier')
    line_albert_scores = []
    for lines in sdf['lyrics_lines']:
        print('scanning lines with bert')
        albert_score_l = []
        for line in lines:
            albert_pred_raw = bert_cls(line)
            albert_pred_raw = albert_pred_raw[0]
            albert_l = [score_label['score'] for score_label in albert_pred_raw]
            albert_score_l.append(albert_l)
        print('done scanning all lines of each song')
        ly_albert_arr = np.array(albert_score_l)
        albert_score = np.sum(ly_albert_arr, axis=0)
        line_albert_scores.append(albert_score)
        print('adding all lines bert scores of a song')
    print('done bert for loops')
    sdf['line_albert_score'] = line_albert_scores
    return sdf['line_albert_score']