"""
Check if all Text-To-Speech are producing valid audio.
We verify the content using a good STT model
"""

import os
import pathlib

import pytest
from livekit import agents
from livekit.agents import tokenize
from livekit.agents.utils import AudioBuffer, merge_frames
from livekit.plugins import silero
from pydub import AudioSegment
from pydub.playback import play
from livekit.agents import JobContext, WorkerOptions, cli, tts, tokenize

TEST_AUDIO_SYNTHESIZE = "В недрах тундры выдры в г+етрах т+ырят в вёдра +ядра к+едров."

@pytest.mark.usefixtures("job_process")
async def test_synthesize():
    tts = agents.tts.StreamAdapter(
        tts=silero.tts.TTS(model='silero_tts', model_id='v4_ru', language='ru', sample_rate=8000, speaker='aidar'),
        sentence_tokenizer=tokenize.basic.SentenceTokenizer(), )

    frames = []
    async for audio in tts.synthesize(TEST_AUDIO_SYNTHESIZE):
        frames.append(audio.frame)

    merged_frame = merge_frames(frames)
    # Преобразование фреймов в AudioSegment
    audio_data = b''.join([frame.data for frame in frames])
    audio_segment = AudioSegment(
        data=audio_data,
        sample_width=4,
        frame_rate=tts.sample_rate,
        channels=tts.num_channels
    )

    # Проигрывание аудио
    play(audio_segment)