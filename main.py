import os
import re
import time
import glob
import json
import random
import requests
import subprocess
from dataclasses import dataclass, field
from typing import List
#from moviepy import *
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
MUBERT_API_KEY = os.getenv("MUBERT_API_KEY", None)  # None means skip AI soundtrack

# For clarity, we assume these are set as environment vars or replaced inline.
# Example in terminal:
# export YOUTUBE_API_KEY="sk-xxx..."


# -----------------------------
#  DATA CLASSES
# -----------------------------
@dataclass
class SubtitleLine:
    start: float
    end: float
    text: str
    score: float = field(default=0.0)


@dataclass
class ClipInstruction:
    source_video: str
    start: float
    end: float
    volume: float = 1.0


@dataclass
class FilmInstructions:
    """Declarative structure describing the final film."""
    clips: List[ClipInstruction] = field(default_factory=list)
    soundtrack_path: str = ""  # e.g., "mubert_track.mp3"
    narration_path: str = ""   # if you want voiceover
    output_filename: str = "essay_film.mp4"


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


# -----------------------------
#  2. PARSE SUBTITLES
# -----------------------------
def parse_subtitle_file(subtitle_path):
    """
    Parse a .vtt or .srt file into a list of SubtitleLine.
    We'll do a naive regex approach for time ranges.
    """
    with open(subtitle_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split blocks by double newlines
    blocks = re.split(r"\n\s*\n", content.strip())
    lines = []
    for block in blocks:
        block = block.strip()
        match = re.search(
            r"(\d{2}:\d{2}:\d{2}[\.,]\d{1,3})\s*-->\s*(\d{2}:\d{2}:\d{2}[\.,]\d{1,3})",
            block,
        )
        if not match:
            continue
        start_str, end_str = match.groups()
        start_sec = timestamp_to_seconds(start_str)
        end_sec = timestamp_to_seconds(end_str)

        # Remove that line from block to get text
        text_block = re.sub(
            r"(\d{2}:\d{2}:\d{2}[\.,]\d{1,3})\s*-->\s*(\d{2}:\d{2}:\d{2}[\.,]\d{1,3})",
            "",
            block,
        )
        text_block = re.sub(r"^\d+\s*$", "", text_block, flags=re.MULTILINE)  # SRT IDs

        text_block = text_block.replace("\n", " ").strip()
        if text_block:
            lines.append(SubtitleLine(start_sec, end_sec, text_block, 0.0))

    return lines


def timestamp_to_seconds(ts):
    """Convert HH:MM:SS,mmm or HH:MM:SS.mmm to float seconds."""
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    if "." in s:
        s, ms = s.split(".")
        return int(h) * 3600 + int(m) * 60 + float(s) + float("0." + ms)
    else:
        return int(h) * 3600 + int(m) * 60 + float(s)


# -----------------------------
#  3. SCORE SUBTITLES w/ OPENAI (Optional)
# -----------------------------
def score_subtitles_with_openai(sub_lines: List[SubtitleLine]):
    """
    Use ChatCompletion to give each subtitle line a "interesting" score [1..10].
    Caches scores in 'line_scores_cache.json' so that subsequent runs skip already scored lines.
    """
    if not OPENAI_API_KEY:
        print("No OPENAI_API_KEY provided, skipping AI-based subtitle scoring.")
        return sub_lines  # all remain score=0

    cache_file = "line_scores_cache.json"
    # Load existing cache if available
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except FileNotFoundError:
        cache = {}

    # Helper to generate a unique key for a subtitle line
    def get_key(sl: SubtitleLine):
        return f"{sl.text.strip()}_{sl.start}"

    # Assign cached scores and collect lines that need scoring
    to_score = []
    for sl in sub_lines:
        key = get_key(sl)
        if key in cache:
            sl.score = cache[key]
        else:
            to_score.append(sl)

    if not to_score:
        print("All subtitle lines have cached scores, skipping API call.")
        return sub_lines

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    chunk_size = 15  # lines per prompt
    i = 0
    while i < len(to_score):
        batch = to_score[i: i + chunk_size]
        print(f"Scoring subtitles for batch {i+1} to {min(i+chunk_size, len(to_score))} out of {len(to_score)}")
        user_content = "Rate each snippet from 1 to 10 for how interesting it is:\n"
        for idx, sl in enumerate(batch):
            user_content += f"Snippet {idx+1}: {sl.text}\n"

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides numeric ratings.",
            },
            {"role": "user", "content": user_content},
        ]
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0,
                max_tokens=300
            )
            reply = response.choices[0].message.content.strip()
            print("Received reply from OpenAI API:", reply)
            # parse lines: "Snippet 1: 7"
            lines = reply.splitlines()
            scores = []
            for line in lines:
                m = re.search(r"Snippet\s+(\d+)\s*:\s*(\d+)", line)
                if m:
                    snippet_idx = int(m.group(1)) - 1
                    rating = float(m.group(2))
                    scores.append((snippet_idx, rating))
            for (snippet_idx, rating) in scores:
                if 0 <= snippet_idx < len(batch):
                    batch[snippet_idx].score = rating
                    # Update cache for this line
                    key = get_key(batch[snippet_idx])
                    cache[key] = rating
        except Exception as e:
            print(f"OpenAI API error: {e}")

        i += chunk_size

    # Save updated cache
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        print(f"Updated line scores cache saved to {cache_file}")
    except Exception as e:
        print(f"Error saving cache: {e}")

    return sub_lines


# -----------------------------
#  4. SELECT TOP MOMENTS
# -----------------------------
def select_top_moments(sub_lines: List[SubtitleLine], top_n=3):
    """
    Return the top N lines by 'score' (descending).
    """
    sorted_lines = sorted(sub_lines, key=lambda x: x.score, reverse=True)
    return sorted_lines[:top_n]


# -----------------------------
#  5. GENERATE AI SOUNDTRACK (Optional, e.g. MUBERT)
# -----------------------------
def generate_mubert_track(duration=60, style="abstract_electronic"):
    """
    Demo Mubert-like call. Adjust to match real Mubert endpoints/payloads.
    Returns a local filename for the downloaded track or None if failed.
    """
    if not MUBERT_API_KEY:
        print("No MUBERT_API_KEY provided, skipping AI soundtrack generation.")
        return None

    # Pseudocode: you must adapt to their real API
    # Let's pretend "preset" param is the style, and "duration" is the length
    url = "https://api.mubert.com/v2/track"
    payload = {
        "apiKey": MUBERT_API_KEY,
        "preset": style,
        "duration": duration,
    }
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Suppose data["data"]["audio"] is a direct link to MP3
        audio_url = data.get("data", {}).get("audio")
        if audio_url:
            audio_filename = f"mubert_{style}_{duration}s.mp3"
            download_file(audio_url, audio_filename)
            return audio_filename
    except Exception as e:
        print(f"Mubert API error: {e}")

    return None


def download_file(url, out_filename):
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(out_filename, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)


# -----------------------------
#  6. BUILD DECLARATIVE FILM INSTRUCTIONS
# -----------------------------
def build_film_instructions(
    video_clips_info,
    soundtrack_path="",
    output_filename="essay_film.mp4"
) -> FilmInstructions:
    """
    video_clips_info is a list of (source_video, start_sec, end_sec).
    Create a FilmInstructions object describing each subclip.
    """
    instructions = FilmInstructions(output_filename=output_filename)
    for (src, st, ed) in video_clips_info:
        instructions.clips.append(ClipInstruction(source_video=src, start=st, end=ed))
    instructions.soundtrack_path = soundtrack_path
    return instructions


def save_film_instructions_to_json(
    instructions: FilmInstructions, json_file="essay_film_instructions.json"
):
    data = {
        "clips": [
            {
                "source_video": c.source_video,
                "start": c.start,
                "end": c.end,
                "volume": c.volume,
            }
            for c in instructions.clips
        ],
        "soundtrack_path": instructions.soundtrack_path,
        "narration_path": instructions.narration_path,
        "output_filename": instructions.output_filename,
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved film instructions to {json_file}")


# -----------------------------
#  7. RENDER FINAL FILM FROM INSTRUCTIONS
# -----------------------------
def load_film_instructions_from_json(json_file="essay_film_instructions.json") -> FilmInstructions:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    fi = FilmInstructions(
        clips=[],
        soundtrack_path=data.get("soundtrack_path", ""),
        narration_path=data.get("narration_path", ""),
        output_filename=data.get("output_filename", "essay_film.mp4"),
    )
    for c in data.get("clips", []):
        fi.clips.append(
            ClipInstruction(
                source_video=c["source_video"],
                start=c["start"],
                end=c["end"],
                volume=c.get("volume", 1.0),
            )
        )
    return fi


def render_film_from_instructions(
    instructions: FilmInstructions,
    final_file=None
):
    subclips = []
    for clip_info in instructions.clips:
        try:
            vc = VideoFileClip(clip_info.source_video).subclip(clip_info.start, clip_info.end)
            vc = vc.volumex(clip_info.volume)
            # Apply a random static zoom effect
            A = random.uniform(0.15, 0.80)  # Zoom variation between 5% and 20%
            zoom_factor = 1 + A
            w, h = vc.w, vc.h
            x1_val = int((w - (w / zoom_factor)) / 2)
            x2_val = int((w + (w / zoom_factor)) / 2)
            y1_val = int((h - (h / zoom_factor)) / 2)
            y2_val = int((h + (h / zoom_factor)) / 2)
            from moviepy.video.fx.all import crop
            vc = crop(vc, x1=x1_val, x2=x2_val, y1=y1_val, y2=y2_val)
            vc = vc.resize((w, h))

            from moviepy.video.fx.all import colorx
            # ... after cropping and resizing
            vc = colorx(vc, 1.2)  # Increase color saturation by 20%
            from moviepy.video.fx.all import lum_contrast

            # Adjust luminance, contrast, and optionally saturation.
            #vc = lum_contrast(vc, lum=0, contrast=50, contrast_thr=128)
            vc = vc.fl_image(glitch_effect)
            
            # Add subtitles if available
            sub_text = get_subtitle_text_for_clip(clip_info.source_video, clip_info.start, clip_info.end)
            if sub_text:
                print(f"Adding subtitles to clip from {clip_info.source_video}")
                vc = add_subtitles_to_clip(vc, sub_text)
            
            subclips.append(vc)
        except Exception as e:
            print(f"Error loading subclip from {clip_info.source_video}: {e}")

    if not subclips:
        print("No subclips to render.")
        return

    if subclips:
        print("Applying fade in to the first clip and fade out to the last clip.")
        from moviepy.video.fx.all import fadein, fadeout
        subclips[0] = fadein(subclips[0], 1)  # 1-second fade in for the first clip
        subclips[-1] = fadeout(subclips[-1], 1)  # 1-second fade out for the last clip

    final = concatenate_videoclips(subclips, method="compose")

    # Preserve original clip audio if available
    original_audio = final.audio

    # Add additional audio tracks: soundtrack and narration
    audio_clips = []
    if original_audio:
        print("Preserving original clip audio at reduced volume.")
        reduced_audio = original_audio.volumex(0.3)  # Lower the volume to 50%
        #reduced_audio = original_audio
        audio_clips.append(reduced_audio)
    if instructions.soundtrack_path and os.path.exists(instructions.soundtrack_path):
        print(f"Adding soundtrack: {instructions.soundtrack_path}")
        music = AudioFileClip(instructions.soundtrack_path)
        if music.duration > final.duration:
            music = music.subclip(0, final.duration)
        audio_clips.append(music)
    if instructions.narration_path and os.path.exists(instructions.narration_path):
        print(f"Adding narration audio: {instructions.narration_path}")
        narration = AudioFileClip(instructions.narration_path)
        if narration.duration > final.duration:
            narration = narration.subclip(0, final.duration)
        audio_clips.append(narration)
    if audio_clips:
        final_audio = CompositeAudioClip(audio_clips)
        final = final.set_audio(final_audio)

    output_name = final_file or instructions.output_filename
    final.write_videofile(output_name, codec="libx264", audio_codec="aac")

    # cleanup
    for c in subclips:
        c.close()
    final.close()
    print(f"Final film saved as {output_name}")


# -----------------------------
#  MAIN DEMO
# -----------------------------
def generate_narration_script(clips_info):
    """
    Use OpenAI's ChatCompletion to generate a narration script for the film,
    based on the provided clips information.
    """
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = "Write a script/narration for a found footage essay film inspired by hito steyerl and harun farocki. David Jhave Johnston. The film is composed of the following clips:\n"
    for (vid_file, start, end) in clips_info:
        # Get transcription lines for the clip
        transcription = get_subtitle_text_for_clip(vid_file, start, end)
        if transcription:
            #prompt += f"- Clip from {vid_file} starting at {start:.2f}s to {end:.2f}s. Transcription: {transcription}\n"
            prompt += "{transcription}\n"
        else:
            prompt += ""
            #prompt += f"- Clip from {vid_file} starting at {start:.2f}s to {end:.2f}s\n"
    #prompt += "\nThe narration should merge these transcriptions into the style: Postmodern poet, Hito Steyrel whose works splice fragmented memories, glitching realities, and scraped digital detritus into jagged, data-driven verse. Explore the tension between human identity and algorithmic distortion, where poetry becomes both a database and a broken mirror, but in a roundabout way without directly using technical lingos or only if they are extremely specific. Use profanity."
    prompt += "\nThe narration should merge these transcriptions into the style: whose works splice fragmented memories, glitching realities, and scraped digital detritus into jagged, data-driven verse. David Jhave Johnston. Use profanity."
    promptt = """Also incorporate sample text from
    ## Emergence of Dark Gothic MAGA

In the middle of a grim election cycle—though nobody could recall precisely which year, for time was misaligned—Elon Musk appeared on a broadcast, wearing a black MAGA hat that seemed to flicker like broken starlight under studio LEDs. He announced a new faction—Dark Gothic MAGA—his tone sliding between venture-capital rhetoric and the theatrical mania of an end-times cult leader. Onlookers and supporters alike stayed transfixed across their screens.

> “The liquidity rules the world,” he declared, “and we who swim in its dark pools shall rule the future, devour the future.”
> 

Those words, woven from panic and prophecy, flickered across every cable channel and smartphone across the world. Soon they were repeated like scripture in the corridors of power, spurring an ominous carnival of opportunists and true believers alike.

José Saramago once wrote, “Inside us there is something that has no name, that something is what we are.” Here, that “something” took the shape of an insatiable hunger for new frontiers: real estate, cryptocurrency, the intangible illusions of an economy whose valences few could understand. They called it “progress,” but if you closed your eyes at night, you might still see the black hat floating in the dark.

In this new Dark Gothic cosmos, the cost of living soared to impossible heights: government presses running ceaselessly “to keep the markets afloat.” In the aftermath, each city block became a carnival of unsettled debts. People sold slices of their tomorrows to buy precarious dwellings in towers owned by conglomerates with spectral names like Obsidian Capital. Rumor had it that this spectral entity possessed not merely properties but the intangible essence of entire neighborhoods—five million units from some long-forgotten mortgage crisis, all quietly consumed and repackaged into the game of modern survival.

Participation was inevitable.

One could not simply choose to stand outside the system; education demanded tuition, healthcare demanded insurance, computing required subscriptions, the future demanded the purchase of intangible securities. So, the populace muttered and cursed while bidding on the next round of volatile assets, all to secure a flicker of stability in a world where Dark Gothic MAGA's mantra—precarity, defunding, ghosting, haunting—was etched into daily life. As if invoking Fredric Jameson, the local newspapers concluded: “*It is far easier to imagine the end of the world than the end of this capitalism.*”

## A Peculiar Heir

Whispers swirled of a “child wiz” anointed by this new gothic order, a strange youth half-coded in binary, half-born from the corners of social media memes. They said his eyes carried the reflection of satellights, and that his singular mission was to sever the heads of all who dared question the new dispensation. Grim rumors claimed that devotees of Dark Gothic MAGA rejoiced in the notion of purging dissent, hoping the purge might cleanse them of their own silent fears. Meanwhile, the opposition, if there was any, now shuffled in place like marionettes, lacking any tangible plan to curb the tide. Their manifestos lay scattered, half-finished, at the feet of uninterested peasants. They chanted how debt was a sacrament, subscriptions were a necessary ritual, and intangible securities a sublime miracle of faith.

> “Nothing can help them,” the child wiz was rumored to say in a voice both ancient and eerily out of time. “Their world is built on illusions of distributive equity. Ours is built on unstoppable inevitabilities.”
> 

## An Empire of Falling Stars

Then, one night, the sky cracked open, releasing hundred thousand Starlink satellites in a dazzling, militant formation. They cut across the heavens in silent waves, as if staking celestial claims to every inch of space. The MAGA faithful pointed upwards in triumph.

“They will see us from the margins now,” they chanted. “They will see and tremble.”

In the half-light of these mechanical constellations, you could hear the anxious murmurs from the edges of Utopia—the same hush once heard in fishermen’s coves and hidden communes, where small gatherings had dared to believe life might be governed by fellowship and little wonders. Now, confronted by this airborne armada, they felt the chill of a creeping enclosure, as if someone had built a fence around the sky itself.

Yet in the whispered corners of those margins, an echo from the Magical Marxists lingered: their half-mystical, half-rational incantations proclaiming that no empire—no matter how colossal or well-funded—could fully extinguish the seeds of another world. One was reminded of Benjamin’s old warning, “*There is no document of civilization which is not at the same time a document of barbarism.” U*nder black umbrellas, small assemblies passed around contraband manuscripts brimming with new forms of solidarity. Recipes for free bread and songs for futures that cost everything mingled in defiance.

## Edge of a Dream

The mesmerizing power of Dark Gothic MAGA relied on its ability to turn desperation into spectacle. It fed on the lust for a safe retirement, for a guaranteed place to live, for an education that wouldn’t shatter one’s finances—all real needs in a precarious age. But it transformed these desires into a high-stakes gamble in which the house always won. Corporations, markets, the devouring reach of intangible capital: they conspired to make you believe *there is no other way.* And so countless people, pinned by the weight of necessity, felt compelled to play the game, hoping luck might favor them—just once.

Back in the parade stands, the child wiz raised a ceremonial blade high, as if to punctuate a final flourish in Dark Gothic MAGA’s opera. Applause erupted from the crowd below, though for some it was laced with dread. From the light eminating from the phones, a hush spread, like a secret about to be revealed. A forest of phone screens glowed brighter, then flickered. Star-lights shimmered on the horizon, creating fleeting silhouettes—illusions, or glimpses of a different possibility?

For a moment, time seemed to pause. One could almost hear the ocean humming an ancient lullaby, a testament to the persistence of wonders. In that lull, a faint voice reminded us:

“Nothing has changed. Everything can change.”

We may now reside under satellite-laden skies, at the mercy of monstrous landlords, in a system that demands we kneel or starve. Yet the seeds of a different tomorrow remain planted in the margins—where people still dare to inquire, craft, and conspire in the name of a world that forever slips from the grasp of those who would claim it.
    """
    messages = [
        {"role": "system", "content": "You are a creative script writer."},
        {"role": "user", "content": prompt},
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            #model="o3-mini",
            messages=messages,
            max_tokens=5000
        )
        script = response.choices[0].message.content.strip()
        print("Generated narration script:", script)
        return script
    except Exception as e:
        print("Error generating narration script:", e)
        return ""


def generate_narration_audio(script, lang="en", out_filename="narration.mp3"):
    """
    Convert the narration script to an audio file using gTTS.
    Requires gTTS library: pip install gtts
    """
    try:
        from gtts import gTTS
        tts = gTTS(text=script, lang=lang)
        tts.save(out_filename)
        print(f"Narration audio saved as {out_filename}")
        return out_filename
    except Exception as e:
        print("Error generating narration audio:", e)
        return ""

def get_subtitle_text_for_clip(video_file, clip_start, clip_end):
    """
    Given a video file and a clip time range, attempt to find the associated subtitle file,
    parse it, and return a string of subtitle lines that fall entirely within the clip range.
    """
    import os
    base = os.path.basename(video_file)
    # Assuming the file naming follows the pattern: "footage/<title>_<video_id>.mp4"
    parts = base.rsplit("_", 1)
    if len(parts) < 2:
        return ""
    video_id_with_ext = parts[-1]
    video_id = video_id_with_ext.split(".")[0]
    sub_file = find_file_with_id(video_id, ".vtt")
    if not sub_file:
        return ""
    lines = parse_subtitle_file(sub_file)
    # Filter subtitle lines that fall within the clip time range
    clip_lines = [line.text for line in lines if line.start >= clip_start and line.end <= clip_end]
    return "\n".join(clip_lines)


def add_subtitles_to_clip(clip, subtitle_text):
    """
    Given a video clip and subtitle text, create a TextClip and overlay it on the video.
    Adjust fontsize and positioning as needed.
    """
    from moviepy.editor import TextClip, CompositeVideoClip
    # Create a TextClip that will act as the subtitle overlay.
    #txt_clip = TextClip(subtitle_text, fontsize=24, color='white', method='caption', size=(clip.w - 20, None))
    # Using pillow method with a semi-transparent background to improve visibility.
    txt_clip = TextClip(
        subtitle_text, 
        fontsize=3, 
        color='white', 
        method='caption',
        bg_color='black',  # Add background to improve visibility
    )
    txt_clip = txt_clip.set_duration(clip.duration).set_position(('center', 'bottom'))
    return CompositeVideoClip([clip, txt_clip])

def glitch_effect(frame):
    import numpy as np
    import random
    # 10% chance to apply a glitch effect on this frame
    if random.random() < 0.1:
        offset = random.randint(-15, 15)
        # Shift the frame horizontally
        frame = np.roll(frame, offset, axis=1)
        # Optionally, add some random noise
        noise = np.random.randint(0, 30, frame.shape, dtype='uint8')
        frame = np.clip(frame + noise, 0, 255)
    return frame

def main():
    parser = argparse.ArgumentParser(description="Generate experimental essay film from YouTube videos.")
    parser.add_argument("--query", type=str, default="America", help="Search query for YouTube videos")
    parser.add_argument("--max_results", type=int, default=3, help="Max results to retrieve from YouTube")
    parser.add_argument("--randomize", action="store_true", help="Randomize clip order")
    parser.add_argument("--output_folder", type=str, default="output", help="Folder to store final output")
    parser.add_argument("--output_prefix", type=str, default="experimental_essay_film", help="Prefix for output file")
    parser.add_argument("--min_clips", type=int, default=5, help="Minimum number of subclips per video")
    parser.add_argument("--max_clips", type=int, default=15, help="Maximum number of subclips per video")
    parser.add_argument("--min_duration", type=float, default=0.1, help="Minimum subclip duration in seconds")
    parser.add_argument("--max_duration", type=float, default=4.0, help="Maximum subclip duration in seconds")
    parser.add_argument("--narrate", action="store_true", help="Generate narration for the film")
    args = parser.parse_args()

    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    # Generate final output filename with current date-time appended
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output_filename = os.path.join(args.output_folder, f"{args.output_prefix}_{now_str}.mp4")

    # 1. Search YouTube: allow comma-separated queries
    queries = [q.strip() for q in args.query.split(',')]
    all_found_videos = []
    for query in queries:
        print(f"Searching YouTube for '{query}'...")
        found_videos = search_youtube_videos(query, max_results=args.max_results)
        all_found_videos.extend(found_videos)

    if not all_found_videos:
        print("No videos found for any query, exiting.")
        return

    # 2. Download each video + subs
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

    # 3. Generate random subclips for each video
    subclip_info = []
    for vid_file in downloaded:
        try:
            clip = VideoFileClip(vid_file)
            duration = clip.duration
            clip.close()
        except Exception as e:
            print(f"Error loading video {vid_file}: {e}")
            continue
        num_clips = random.randint(args.min_clips, args.max_clips)
        for _ in range(num_clips):
            clip_duration = random.uniform(args.min_duration, args.max_duration)
            if duration <= clip_duration:
                start_sec = 0
                end_sec = duration
            else:
                start_sec = random.uniform(0, duration - clip_duration)
                end_sec = start_sec + clip_duration
            subclip_info.append((vid_file, start_sec, end_sec))

    if not subclip_info:
        print("No interesting subclips found, let's just skip montage.")
        return

    # Optionally randomize clip order if flag provided
    if args.randomize:
        random.shuffle(subclip_info)

    # 4. Generate an AI soundtrack (optional)
    soundtrack_file = None
    if MUBERT_API_KEY:
        soundtrack_file = generate_mubert_track(duration=60, style="abstract_electronic")
    else:
        print("No MUBERT_API_KEY, skipping soundtrack generation.")

    # 5. Build film instructions (montage)
    instructions = build_film_instructions(
        subclip_info, 
        soundtrack_path=soundtrack_file or "", 
        output_filename=final_output_filename
    )
    # Optionally generate narration if --narrate is specified
    if args.narrate:
        print("Generating narration script...")
        narration_script = generate_narration_script(subclip_info)
        if narration_script:
            # Save the narration script with query and datetime
            narration_text_filename = os.path.join(args.output_folder, f"narration_{args.query.replace(' ', '_')}_{now_str}.txt")
            with open(narration_text_filename, "w", encoding="utf-8") as f:
                f.write(narration_script)
            print(f"Narration script saved as {narration_text_filename}")
            
            # Save narration audio with the same naming format
            narration_audio_filename = os.path.join(args.output_folder, f"narration_{args.query.replace(' ', '_')}_{now_str}.mp3")
            narration_audio_path = generate_narration_audio(narration_script, out_filename=narration_audio_filename)
            if narration_audio_path:
                print(f"Narration audio generated at {narration_audio_path}")
                instructions.narration_path = narration_audio_path
            else:
                print("Failed to generate narration audio.")
        else:
            print("Narration script generation failed.")
 
    # 6. Save instructions to JSON for manual editing
    save_film_instructions_to_json(instructions, "essay_film_instructions.json")

    # 7. Render final film from the instructions
    final_instructions = load_film_instructions_from_json("essay_film_instructions.json")
    render_film_from_instructions(final_instructions, final_file=final_output_filename)


if __name__ == "__main__":
    main()