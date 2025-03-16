import os
import argparse
import re
import time
import glob
import json
import random
import requests
import subprocess
import immemory.tts
from dataclasses import dataclass, field
from typing import List
from typing import List, Optional
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips
from googleapiclient.discovery import build
from dateutil import tz
from dotenv import load_dotenv
load_dotenv()
try:
    from PIL import Image
    if not hasattr(Image, 'ANTIALIAS'):
        Image.ANTIALIAS = Image.Resampling.LANCZOS
except ImportError:
    pass
import argparse
from datetime import datetime

os.environ["IMAGEMAGICK_BINARY"] = "magick"

# -----------------------------
#  CONFIG / SECRETS
# -----------------------------
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)  # None means no LLM scoring
SPEECHIFY_API_KEY = os.getenv("SPEECHIFY_API_KEY", None)  # None means no LLM scoring
MUBERT_API_KEY = os.getenv("MUBERT_API_KEY", None)  # None means skip AI soundtrack

# -----------------------------
#  1. YOUTUBE DOWNLOAD
# -----------------------------
def search_youtube_videos(query="America", max_results=3):
    """
    Use the YouTube Data API to find video IDs for a given search query.
    Returns a list of (videoId, title).
    """
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    # Example: search for videos within last 24 hours, or remove publishedAfter if you want broader results
    # Here we skip date filters for simplicity—adjust as needed.
    search_response = (
        youtube.search()
        .list(
            q=query,
            part="snippet",
            type="video",
            order="relevance",
            maxResults=max_results,
        )
        .execute()
    )

    items = search_response.get("items", [])
    results = []
    for it in items:
        vid = it["id"]["videoId"]
        title = it["snippet"]["title"]
        results.append((vid, title))
    return results

def download_youtube_video(video_id):
    """
    Download the YouTube video + auto subtitles (if available) via yt-dlp.
    Output files typically: <title>.mp4 and <title>.*.vtt
    Returns the main .mp4 filename if successful, else None.
    """
    # We'll call yt-dlp via subprocess. Make sure yt-dlp is installed: `pip install yt-dlp`
    # Using '--write-auto-subs' to get auto-generated subs if exist.
    # Using '--sub-lang en' as an example, adjust as needed.
    # Using '--output' to produce a consistent naming pattern with video ID for reference.
    # CAREFUL: Some titles contain special chars. We'll do best to keep it safe.

    out_template = f"footage/%(title)s_{video_id}.%(ext)s"
    cmd = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "--write-auto-subs",
        "--sub-lang", "en",
        "--output", out_template,
        "--format", "mp4/best",
        "--merge-output-format", "mp4",
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading video {video_id}: {e}")
        return None

    # We guess the main .mp4 name by scanning the directory for something containing video_id
    downloaded_mp4 = find_file_with_id(video_id, ".mp4")
    return downloaded_mp4

def find_file_with_id(video_id, extension=".mp4"):
    """Helper to find a file that contains video_id and extension.
    Searches in the 'footage' folder first, then in the current directory."""
    candidates = glob.glob(f"footage/*{video_id}*{extension}")
    if not candidates:
        candidates = glob.glob(f"*{video_id}*{extension}")
    if candidates:
        return candidates[0]
    return None

def expand_keyword_with_gpt(keyword: str, how_many: int = 3) -> List[str]:
    """Use GPT to generate multiple search queries from a single keyword."""
    if not OPENAI_API_KEY:
        print("No OPENAI_API_KEY provided, skipping GPT-based query expansion.")
        return [keyword]

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_message = "You are a creative brainstorming assistant."
    user_message = f"""Expand the following keyword into {how_many} different, short search queries for YouTube:
Keyword: {keyword}

Focus on different angles or 'vectors' (e.g., aesthetic, historical, critical). 
Simply return each query as a separate line without explanation."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    print(user_message)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=200,
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()
        # Parse lines from the response
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        print(lines)
        return lines[0:how_many]
    except Exception as e:
        print("Error expanding keyword with GPT:", e)
        return [keyword]

# -----------------------------
#  2. PARSE SCREENPLAY 
# -----------------------------
class MyBeat:
    def __init__(self):
        self.actions = []     # type: List[str]
        self.dialogues = []   # type: List[(character, line)]

class MySubscene:
    def __init__(self):
        self.beats = []       # type: List[MyBeat]

class MyScene:
    def __init__(self, heading: str, scene_number: Optional[str] = None):
        self.heading = heading
        self.scene_number = scene_number
        self.subscenes = []   # type: List[MySubscene]

class MyScript:
    def __init__(self):
        self.scenes = []      # type: List[MyScene]

class ScriptBuilder:
    def __init__(self):
        self.script = MyScript()

        self.current_scene = None    # type: Optional[MyScene]
        self.current_subscene = None # type: Optional[MySubscene]
        self.current_beat = None     # type: Optional[MyBeat]

        # A naive check for transitions that trigger new subscenes.
        self.transition_regex = re.compile(r'^(CUT TO:|LATER|FLASHBACK|END FLASHBACK|FADE TO BLACK|SMASH CUT TO:).*', re.IGNORECASE)

    def start_new_scene(self, heading: str, scene_number: Optional[str]):
        """Create a new scene and subscene+beat chain."""
        self.current_scene = MyScene(heading, scene_number)
        self.script.scenes.append(self.current_scene)

        self.current_subscene = MySubscene()
        self.current_scene.subscenes.append(self.current_subscene)

        self.current_beat = MyBeat()
        self.current_subscene.beats.append(self.current_beat)

    def start_new_subscene(self):
        """Create a new subscene and new beat."""
        if not self.current_scene:
            # If we haven't started a scene yet, create a dummy one
            self.start_new_scene("UNTITLED SCENE", None)
            return

        self.current_subscene = MySubscene()
        self.current_scene.subscenes.append(self.current_subscene)

        self.current_beat = MyBeat()
        self.current_subscene.beats.append(self.current_beat)

    def ensure_beat_exists(self):
        """Make sure we have a current beat to store actions/dialogues in."""
        if not self.current_beat:
            # If no subscene or beat, create them now
            if not self.current_subscene:
                self.start_new_subscene()
            else:
                self.current_beat = MyBeat()
                self.current_subscene.beats.append(self.current_beat)

    # ---- Callback Methods ----

    def on_scene_heading(self, text: str, scene_number: str):
        # Start a new scene (and automatically new subscene+beat)
        self.start_new_scene(text, scene_number if scene_number else None)

    def on_action(self, text: str):
        self.ensure_beat_exists()
        self.current_beat.actions.append(text)
    
    def _store_last_dialogue_info(self, full_character, line, new_character_line=False):
        if not hasattr(self, "_last_char_info"):
            self._last_char_info = {}
        self._last_char_info["char"] = full_character
        self._last_char_info["is_new_line"] = new_character_line

    def on_dialogue(
        self,
        character: str,
        extension: str,
        parenthetical: str,
        line: str,
        is_dual_dialogue: bool
    ):
        """
        Merge lines that appear to be a continuation of the same typed block
        (i.e. same character, text doesn't appear to start a new block),
        else start a new dialogue entry.
        """
        self.ensure_beat_exists()

        # Combine extension/parenthetical if needed
        full_character = character
        if extension:
            full_character += f" ({extension})"
        if parenthetical:
            full_character += f" [{parenthetical}]"

        line_stripped = line.strip()

        # (CONT'D) check: we treat it as new block if found
        import re

        # This pattern allows either a straight or curly apostrophe after "CONT"
        contd_pattern = re.compile(r'\(CONT[\'’]?D\)', re.IGNORECASE)
        is_contd = bool(contd_pattern.search(full_character) or contd_pattern.search(line_stripped))

        # Heuristic: If line_stripped looks like uppercase (≥2 letters) or starts with '(',
        # it might be a new typed line. We'll detect that here:
        # e.g., "NARRATOR (V.O.)" or "NARRATOR (V.O.) (CONT'D)"
        # or user typed a brand-new chunk of dialogue. 
        # This is a naive check—tweak to your preference.
        starts_like_new_block = False
        if line_stripped:
            # If line is something like "MORE LINES" in uppercase, we treat as new block
            # or if it starts with "(" (like "(whispering)")
            if re.match(r'^[A-Z0-9]{2}', line_stripped) or line_stripped.startswith("("):
                starts_like_new_block = True

        # If there's no last dialogue at all, just start a new block
        if not self.current_beat.dialogues:
            self.current_beat.dialogues.append((full_character, line))
            self._store_last_dialogue_info(full_character, line, new_character_line=True)
            return

        # Check the last stored dialogue
        last_char, last_line = self.current_beat.dialogues[-1]

        last_char_info = getattr(self, "_last_char_info", None)
        if not last_char_info:
            # If no info, define some
            last_char_info = {"char": None, "is_new_line": True}
            setattr(self, "_last_char_info", last_char_info)

        # Decide if we should MERGE or NEW BLOCK
        # We'll do so if same 'full_character', not (CONT'D),
        # not uppercase, not starting with '('
        should_merge = False
        if last_char == full_character:
            # same char => potential merge
            # only merge if not (CONT'D) and not a "start-like-new-block" line
            if not is_contd and not starts_like_new_block:
                should_merge = True

        if should_merge:
            # Merge into the same block
            merged_line = f"{last_line} {line_stripped}".strip()
            self.current_beat.dialogues[-1] = (last_char, merged_line)
            last_char_info["is_new_line"] = False
        else:
            # Start a new block
            self.current_beat.dialogues.append((full_character, line))
            last_char_info["is_new_line"] = True

        # Update stored info
        last_char_info["char"] = full_character
        setattr(self, "_last_char_info", last_char_info)

    def on_transition(self, text: str):
        """If text matches our naive transition triggers, start a new subscene."""
        if self.transition_regex.match(text):
            self.start_new_subscene()

    # The rest are optional if you want to handle them:
    def on_lyrics(self, text: str):
        self.ensure_beat_exists()
        self.current_beat.actions.append(f"LYRICS: {text}")

    def on_section(self, text: str, level: int):
        # Could store as a special marker if you want
        pass

    def on_synopsis(self, text: str):
        pass

    def on_page_break(self):
        pass

    def on_title_page(self, entries):
        # e.g., store them or print them
        pass

from fountain_tools.callback_parser import CallbackParser, TitleEntry
from fountain_tools.parser import Parser as FountainParser
from fountain_tools.writer import Writer as FountainWriter
from fountain_tools.fountain import *

def parse_fountain_into_structure(fountain_text: str) -> MyScript:
    # Create parser and builder
    parser = CallbackParser()
    builder = ScriptBuilder()

    # Assign callbacks
    parser.onTitlePage = builder.on_title_page
    parser.onSceneHeading = builder.on_scene_heading
    parser.onAction = builder.on_action
    parser.onDialogue = builder.on_dialogue
    parser.onTransition = builder.on_transition
    parser.onLyrics = builder.on_lyrics
    parser.onSection = builder.on_section
    parser.onSynopsis = builder.on_synopsis
    parser.onPageBreak = builder.on_page_break

    # Parse the text
    parser.add_text(fountain_text)
    parser.finalize()

    # Return the final data structure
    return builder.script

from fountain_tools.parser import Parser as FountainParser
from fountain_tools.writer import Writer as FountainWriter
from fountain_tools.fountain import *

def print_script_structure(my_script: MyScript):
    """
    Print the script's structure:
      - Scenes
        - Subscenes
          - Beats
            - Actions
            - Dialogues
    """
    for s_i, scene in enumerate(my_script.scenes, start=1):
        print(f"SCENE {s_i}: {scene.heading} (Scene #: {scene.scene_number})")
        for sub_i, subscene in enumerate(scene.subscenes, start=1):
            print(f"  SUBSCENE {sub_i}")
            for b_i, beat in enumerate(subscene.beats, start=1):
                print(f"    BEAT {b_i}")
                if beat.actions:
                    print(f"      Actions ({len(beat.actions)})")
                    for a in beat.actions:
                        print(f"        - {a}")
                if beat.dialogues:
                    print(f"      Dialogues ({len(beat.dialogues)})")
                    for char, dlg in beat.dialogues:
                        print(f"        {char}: {dlg}")

def process_all_beats(my_script: MyScript):
    """
    Iterate over every Scene -> Subscene -> Beat in 'my_script'
    and do something with each beat.
    """
    beat_counter = 1
    
    for scene_index, scene in enumerate(my_script.scenes, start=1):
        for subscene_index, subscene in enumerate(scene.subscenes, start=1):
            for beat_index, beat in enumerate(subscene.beats, start=1):
                
                # Construct an output filename (example):
                output_filename = f"scene{scene_index}_sub{subscene_index}_beat{beat_index}.mp4"
                
                print(f"Processing {output_filename}...")
                
                # Call your function that handles the single beat
                # e.g., compile it into a mini-video:
                compile_beat_video(beat, output_filename=output_filename)
                
                beat_counter += 1

    print(f"Done! Processed {beat_counter - 1} beats total.")

def generate_narration_audio_speechify(script, token, voice_id, output_filename="narration.mp3",
                                       audio_format="mp3", language="en-US", model="simba-english"):
    """
    Generate speech audio using the Speechify TTS API and save it as an mp3 file.

    Parameters:
      script (str): The plain text or SSML to be synthesized.
      token (str): Your Speechify API bearer token.
      voice_id (str): The voice identifier to use for synthesis.
      output_filename (str): File name to save the synthesized audio (default "narration.mp3").
      audio_format (str): Format of the output audio (default "mp3").
      language (str): Language code (default "en-US").
      model (str): Synthesis model (e.g., "simba-english").
    
    Returns:
      str: The output filename if successful, or an empty string on failure.
    """
    url = "https://api.sws.speechify.com/v1/audio/speech"
    headers = {
         "Authorization": f"Bearer {token}",
         "Content-Type": "application/json"
    }
    payload = {
         "input": script,
         "voice_id": voice_id,
         "audio_format": audio_format,
         "language": language,
         "model": model
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
         print("Error calling Speechify TTS API:", response.text)
         return ""
    
    data = response.json()
    audio_data = data.get("audio_data")
    if not audio_data:
         print("No audio data received from Speechify TTS API.")
         return ""
    
    import base64
    audio_bytes = base64.b64decode(audio_data)
    with open(output_filename, "wb") as f:
         f.write(audio_bytes)
    print(f"Narration audio saved as {output_filename}")
    return output_filename

def compile_beat_video(
    beat: MyBeat, 
    scene_index: int,
    subscene_index: int,
    beat_index: int,
    output_filename="beat_output.mp4"):
    """
    1) Summarize action lines -> search youtube for a short clip
    2) Generate TTS for each dialogue
    3) Combine them
    """
    # 1) Summarize action lines
    action_text = " ".join(beat.actions)
    if not action_text.strip():
        action_text = "No action text"

    # 2) Download a short clip from YT
    # We'll keep it to 15 seconds for demonstration
    # If there's no action text, we might just skip
    expanded_queries = expand_keyword_with_gpt(action_text, how_many=1)

    all_found_videos = []
    for q in expanded_queries:
        print(f"Searching YouTube for '{q}'...")
        found_videos = search_youtube_videos(q, max_results=1)
        all_found_videos.extend(found_videos)

    downloaded = []
    for vid, title in all_found_videos:
        print(f"Downloading {vid} : {title}")
        mp4_file = download_youtube_video(vid)
        if mp4_file:
            downloaded.append(mp4_file)
        else:
            print(f"Failed to download video ID {vid}.")

    if not downloaded:
        print("No videos successfully downloaded, exiting.")
        return

    # 3) Generate TTS for dialogues
    dialogue_audios = []
    dialogue_index = 0
    for (character, line) in beat.dialogues:
        dialogue_index += 1
        if not line:
            # If there's no line, skip
            continue
        filename = f"dialogue_{scene_index}_{subscene_index}_{beat_index}_{dialogue_index}.mp3"
        audio_file = generate_narration_audio_speechify(
            line, SPEECHIFY_API_KEY, "c3605b43-d40a-486d-9fe9-e181fe240d6c",
            output_filename=filename)
        dialogue_audios.append(audio_file)

    # 4) Load the video clip and audio files in MoviePy
    video_clip = VideoFileClip(downloaded[0])

    # combine dialogue audios sequentially
    audio_clips = []
    start_time = 0
    for a_file in dialogue_audios:
        # load each dialogue audio
        aclip = AudioFileClip(a_file)
        print(f"Loaded {a_file}, duration={aclip.duration}")
        # place it at 'start_time'
        # If you want them overlapped with the video, you can do a 
        # single composite with offsets. We'll just line them up.
        aclip = aclip.set_start(start_time)
        start_time += aclip.duration + 0.5  # add a short pause
        audio_clips.append(aclip)

    # composite the audio
    if audio_clips:
        composite_audio = CompositeAudioClip(audio_clips)
        print("Composite audio duration =", composite_audio.duration)
        print("Video clip duration before set_duration =", video_clip.duration)
        # If the composite is longer than the video, extend video
        final_duration = max(video_clip.duration, composite_audio.duration)
        video_clip = video_clip.set_duration(final_duration)
        final_audio = video_clip.audio
        if final_audio: 
            # mix with original?
            final_audio = CompositeAudioClip([final_audio, composite_audio])
        else:
            final_audio = composite_audio
        final_video = video_clip.set_audio(final_audio)
    else:
        # no dialogue audio
        final_video = video_clip

    final_video.write_videofile(output_filename, codec="libx264")

    # cleanup
    video_clip.close()
    for a_file in dialogue_audios:
        os.remove(a_file)
    os.remove(downloaded[0])

def process_all_beats(my_script: MyScript):
    """
    Iterate over every Scene -> Subscene -> Beat in 'my_script'
    and do something with each beat.
    """
    beat_counter = 1
    
    for scene_index, scene in enumerate(my_script.scenes, start=1):
        for subscene_index, subscene in enumerate(scene.subscenes, start=1):
            for beat_index, beat in enumerate(subscene.beats, start=1):
                
                # Construct an output filename (example):
                output_filename = f"scene{scene_index}_sub{subscene_index}_beat{beat_index}.mp4"
                
                print(f"Processing {output_filename}...")
                
                # Call your function that handles the single beat
                # e.g., compile it into a mini-video:
                compile_beat_video(
                    beat,
                    scene_index=scene_index,
                    subscene_index=subscene_index,
                    beat_index=beat_index,
                    output_filename=output_filename
                )
                
                beat_counter += 1

    print(f"Done! Processed {beat_counter - 1} beats total.")

def main1():
    from_narration_file = "sample_screenplay.fountain"
    narration_script = ""

    if from_narration_file:
        print(f"Loading narration text from {from_narration_file}...")
        try:
            with open(from_narration_file, "r", encoding="utf-8") as f:
                narration_script = f.read()
        except Exception as e:
            print("Failed to load narration file:", e)
            narration_script = ""

    script = parse_fountain_into_structure(narration_script)
    print(script)
    print_script_structure(script)


def main():
    from_narration_file = "sample_screenplay.fountain"
    narration_script = ""

    if from_narration_file:
        print(f"Loading narration text from {from_narration_file}...")
        try:
            with open(from_narration_file, "r", encoding="utf-8") as f:
                narration_script = f.read()
        except Exception as e:
            print("Failed to load narration file:", e)
            narration_script = ""

    script = parse_fountain_into_structure(narration_script)
    print(script)
    print_script_structure(script)
    input("Press Enter to continue...")
    process_all_beats(script)

if __name__ == "__main__":
    main()
