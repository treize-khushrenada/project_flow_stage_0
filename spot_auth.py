import os
import sys
import json
import spotipy
import webbrowser
import spotipy.util as util
from json.decoder import JSONDecodeError

username = sys.argv[1]

try:
	token = util.prompt_for_user_token(username)
except:
	os.remove(f".cache-{username}")
	token = util.prompt_for_user_token(username)

spot_obj = spotipy.Spotify(auth=token)