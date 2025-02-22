import json
import os

import requests
import yt_dlp as youtube_dl
from deepgram import DeepgramClient, FileSource, PrerecordedOptions
from dotenv import load_dotenv

load_dotenv()

DG_KEY = os.getenv('DEEPGRAM_API_KEY')
SAVE_DIR = '/'.join(os.getcwd().split('/')[:3]) + 'static/speechdir'
AUDIO_PATH = SAVE_DIR + '/audio_cont'
JSON_PATH = SAVE_DIR + '/json_cont'

class AudioProcessor:
    """
    AudioProcessor class for processing audio files.
    """
    deepgram = DeepgramClient(DG_KEY)
    video_path: str = None
    ydl_opts: list = {
        'format': 'best',
        'outtmpl': AUDIO_PATH + '/%(title)s.%(ext)s',
        'noplaylist': True,
        'extract-audio': True,
    }
    def __init__(self, 
                 audio_path: str = None,
                 ydl_opts: dict = ydl_opts,
                 **kwargs):
        """
        Initializes the AudioProcessor.
        
        :param audio_path: str. The path to the audio file.
        """
        super()
        if audio_path is None:
            print("No audio file provided.")
        else:
            self.video_path = audio_path
            self.ydl_opts = ydl_opts
            print(f"Loaded audio file {self.video_path}.")
    
    def process_download(video_url: str = None, 
                         ydl_opts: dict = ydl_opts, 
                         audio_path: str = AUDIO_PATH, 
                         **kwargs):
        #if in SAVE_PATH have another file, delete it to be empty
        if os.path.exists(audio_path):
            for file in os.listdir(audio_path):
                os.remove(audio_path + '/' + file)
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            video_url = info_dict.get("url", None)
            video_title = info_dict.get('title', None)
            # video_length = info_dict.get('duration')
        return video_title

    def transcript(audio_path: str = AUDIO_PATH,
                   json_path: str = JSON_PATH,
                   **kwargs):
        audio_file = audio_path + '/' + video_title + '.mp4'
        json_file = json_path + '/' + video_title + '.json'
        try:
            # Download the audio file from the URL
            if os.path.exists(json_path):
                for file in os.listdir(json_path):
                    os.remove(json_path + '/' + file)

            with open(audio_file, "rb") as file:
                buffer_data = file.read()

            payload: FileSource = {
                "buffer": buffer_data,
            }

            # STEP 2: Configure Deepgram options for audio analysis
            options = PrerecordedOptions(
                model="nova-2",
                smart_format=True,
            )

            # STEP 3: Call the transcribe_file method with the text payload and options
            response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)

            # STEP 4: Write the response JSON to a file
            with open(json_file, "w") as transcript_file:
                transcript_file.write(response.to_json(indent=4))

            print("Transcript JSON file generated successfully.")

        except Exception as e:
            print(f"Exception: {e}")
        
    def transcript(transcription_file: str = None):
        with open(transcription_file, "r") as file:
            final_res = ""
            data = json.load(file)
            result = data['results']['channels'][0]['alternatives'][0]['transcript']
            result = result.split('.')
            for sentence in result:
                final_res += sentence + '. '
                # print(sentence + '.')
            return final_res