"""
media_utils.py
Transcribes audio and video files to text, so the app can accept an
audio clip or a video file (news broadcast, voice memo, etc.) and run
the transcript through the same fake-news classifier.

Requires ffmpeg to be installed on the system (used by pydub under the
hood to decode video/audio formats). On Streamlit Community Cloud, add
a `packages.txt` file to the repo root containing the line "ffmpeg" so
it gets installed automatically at deploy time.

Uses SpeechRecognition's free Google Web Speech API for transcription
(no API key needed) rather than a heavier local model like Whisper --
this keeps the app light enough to run on free-tier hosting, at the
cost of needing an internet connection and being less accurate on
noisy audio.
"""

import os
import tempfile

import speech_recognition as sr
from pydub import AudioSegment

CHUNK_MS = 55_000  # keep chunks under ~1 minute; the free API works best on short segments


def _convert_to_wav(input_path: str) -> str:
    """
    Converts any audio or video file pydub/ffmpeg can read into a mono,
    16kHz WAV file (the format speech_recognition needs), and returns
    the path to that temp WAV file. Works for video files too -- pydub
    calls ffmpeg under the hood, which extracts the audio track automatically.
    """
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)

    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    audio.export(wav_path, format="wav")
    return wav_path


def transcribe_media_file(input_path: str) -> str:
    """
    Transcribes an audio OR video file to text.
    Raises RuntimeError with a clear message on failure (bad file,
    no speech recognized, or no internet reaching the recognition API).
    """
    wav_path = _convert_to_wav(input_path)
    recognizer = sr.Recognizer()

    try:
        audio = AudioSegment.from_wav(wav_path)
    finally:
        pass  # wav_path cleaned up at the end regardless of what happens above

    chunks = [audio[i:i + CHUNK_MS] for i in range(0, len(audio), CHUNK_MS)]
    if not chunks:
        os.remove(wav_path)
        raise RuntimeError("The file appears to contain no audio.")

    transcript_parts = []
    try:
        for idx, chunk in enumerate(chunks):
            fd, chunk_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            chunk.export(chunk_path, format="wav")

            try:
                with sr.AudioFile(chunk_path) as source:
                    audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data)
                    transcript_parts.append(text)
                except sr.UnknownValueError:
                    # This chunk had no recognizable speech (silence, noise, music) -- skip it
                    continue
                except sr.RequestError as e:
                    raise RuntimeError(
                        f"Speech recognition service unreachable: {e}. "
                        "Check your internet connection."
                    ) from e
            finally:
                os.remove(chunk_path)
    finally:
        os.remove(wav_path)

    transcript = " ".join(transcript_parts).strip()
    if not transcript:
        raise RuntimeError(
            "No speech could be recognized in this file. "
            "Try a clearer recording, or paste the text directly instead."
        )
    return transcript
