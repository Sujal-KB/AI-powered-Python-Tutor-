from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
import yt_dlp
import requests


def get_video_id(url):
    if "v=" in url:
        start = url.find("v=") + 2
        end = url.find("&", start)
        if end == -1:
            return url[start:]
        return url[start:]
    return url


# 🔹 Primary method (YouTube Transcript API)
def get_transcript_api(video_id):
    transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en", "hi"])
    return " ".join([i.text for i in transcript])


# 🔹 Fallback method (yt-dlp)
def get_transcript_yt_dlp(video_url):
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "quiet": True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)

            subtitles = info.get("subtitles") or info.get("automatic_captions")

            if subtitles:
                for lang in subtitles:
                    subtitle_url = subtitles[lang][0]["url"]

                    res = requests.get(subtitle_url)
                    return res.text
    except Exception:
        return None

    return None


# 🔥 Final function with fallback
def get_transcript(yt_video_url):
    video_id = get_video_id(yt_video_url)

    # Try primary method
    try:
        return get_transcript_api(video_id)
    except Exception:
        pass

    # Try fallback
    transcript = get_transcript_yt_dlp(yt_video_url)
    if transcript:
        return transcript

    return "Transcript not available for this video."


# 🔹 Text splitter (same as yours)
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.create_documents([text])
    return chunks
