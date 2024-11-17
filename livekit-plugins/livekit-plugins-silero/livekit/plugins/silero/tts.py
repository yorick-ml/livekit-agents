# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
from livekit import rtc
from livekit.agents import tts, utils

from .log import logger

class SileroTTS(tts.TTS):
    def __init__(
        self,
        *,
        repo_or_dir: str = 'snakers4/silero-models',
        model: str = 'silero_tts',
        language: str = 'ru',
        model_id: str = 'v4_ru',
        speaker: str = 'aidar',
        sample_rate: int = 48000,
        device: torch.device = torch.device('cpu'),
    ) -> None:
        """
        Create a new instance of Silero TTS.

        Args:
            repo_or_dir (str): Repository or directory containing the model. Defaults to 'snakers4/silero-models'.
            model (str): Model name. Defaults to 'silero_tts'.
            language (str): Language code. Defaults to 'ru'.
            model_id (str): model_id. Defaults to 'v4_ru'.
            speaker (str): Speaker name. Defaults to 'aidar'.
            sample_rate (int): Sample rate for the output audio. Defaults to 48000.
            device (torch.device): Device to use for inference. Defaults to 'cpu'.
        """

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )

        self._speaker = speaker
        self._device = device
        self._model, self._example_text = torch.hub.load(
            repo_or_dir=repo_or_dir,
            model=model,
            language=language,
            speaker=model_id,
        )
        self._model.to(self._device)

    def synthesize(self, text: str) -> "ChunkedStream":
        return ChunkedStream(self, text, self._speaker, self._device, self._model)

    def stream(self) -> "SynthesizeStream":
        return SynthesizeStream(self, self._speaker, self._device, self._model)


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        tts: SileroTTS,
        text: str,
        speaker: str,
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(tts, text)
        self._speaker = speaker
        self._device = device
        self._model = model

    async def _main_task(self) -> None:
        request_id = utils.shortuuid()
        audio = self._model.apply_tts(
            text=self._input_text,
            speaker=self._speaker,
            sample_rate=self._tts.sample_rate,
        )

        audio_frame = rtc.AudioFrame(
            data=audio.numpy().tobytes(),
            sample_rate=self._tts.sample_rate,
            num_channels=1,
            samples_per_channel=len(audio),
        )
        self._event_ch.send_nowait(
            tts.SynthesizedAudio(
                request_id=request_id,
                frame=audio_frame,
            )
        )


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        tts: SileroTTS,
        speaker: str,
        device: torch.device,
        model: torch.nn.Module,
    ):
        super().__init__(tts)
        self._speaker = speaker
        self._device = device
        self._model = model

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        async for input_text in self._input_ch:
            if isinstance(input_text, str):
                request_id = utils.shortuuid()
                audio = self._model.apply_tts(
                    text=input_text,
                    speaker=self._speaker,
                    sample_rate=self._tts.sample_rate,
                )
                audio_frame = rtc.AudioFrame(
                    data=audio.numpy().tobytes(),
                    sample_rate=self._tts.sample_rate,
                    num_channels=1,
                    samples_per_channel=len(audio),
                )
                self._event_ch.send_nowait(
                    tts.SynthesizedAudio(
                        request_id=request_id,
                        frame=audio_frame,
                    )
                )
