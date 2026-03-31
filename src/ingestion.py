from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_video_id(url):
    if "v=" in url:
        start = url.find("v=") + 2
        end = url.find("&", start)
        if end == -1:
            return url[start:]
        return url[start:end]
    return url


def get_transcript(video_url):
    video_id = get_video_id(video_url)

    try:
        transcript = YouTubeTranscriptApi().fetch(video_id,languages=['en','hi'])
        return " ".join([i.text for i in transcript])
    except Exception:
        return "Transcript not available for this video."


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.create_documents([text])
    return chunks
