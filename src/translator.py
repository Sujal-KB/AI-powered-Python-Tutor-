from deep_translator import GoogleTranslator
import time


def translate_text(text: str) -> str:
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        print("Translation error:", e)
        return text


def translate_chunks(chunks):
    translated_chunks = []

    for i, chunk in enumerate(chunks):
        text = chunk.page_content.strip()

        if not text:
            continue

        try:
            print(f"Translating chunk {i+1}...")
            translated_text = translate_text(text[:4000])
            time.sleep(0.5)
        except Exception as e:
            print(f"Error at chunk {i+1}: {e}")
            translated_text = text

        chunk.page_content = translated_text
        translated_chunks.append(chunk)

    return translated_chunks