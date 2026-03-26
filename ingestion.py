from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_transcript(yt_video_id):
  
  def get_video_id(url):
    
    if "v=" in url:
        start = url.find("v=") + 2
        end = url.find("&", start)
        if end == -1:
            return url[start:]
        return url[start:end]
    
  transcript = YouTubeTranscriptApi().fetch(
      get_video_id(yt_video_id),
      languages=["en",'hi']
  )

  full_text = " ".join([i.text for i in transcript])
  return full_text


def split_text(text):
   splitter=RecursiveCharacterTextSplitter(
      chunk_size=800,
      chunk_overlap=100
   )

   chunks=splitter.create_documents([text])

   return chunks