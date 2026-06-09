"""Compatibility wrapper for Sarvam TTS MP3 responses.

The LiveKit Sarvam plugin currently labels Sarvam's returned bytes as WAV, but
Sarvam is returning MPEG audio bytes. This wrapper keeps Sarvam as the provider
and lets LiveKit decode the bytes with its MP3 decoder.
"""

from __future__ import annotations

import asyncio
import base64

import aiohttp
from livekit.agents import tts
from livekit.agents._exceptions import APIConnectionError, APIStatusError, APITimeoutError
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.plugins.sarvam import tts as sarvam_tts


class SarvamMpegChunkedStream(sarvam_tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        payload = {
            "target_language_code": self._opts.target_language_code,
            "text": self._input_text,
            "speaker": self._opts.speaker,
            "pace": self._opts.pace,
            "speech_sample_rate": self._opts.speech_sample_rate,
            "model": self._opts.model,
            "output_audio_bitrate": self._opts.output_audio_bitrate,
            "min_buffer_size": self._opts.min_buffer_size,
            "max_chunk_length": self._opts.max_chunk_length,
        }
        if self._opts.model == "bulbul:v2":
            payload["pitch"] = self._opts.pitch
            payload["loudness"] = self._opts.loudness
            payload["enable_preprocessing"] = self._opts.enable_preprocessing
        if self._opts.model in ("bulbul:v3", "bulbul:v3-beta"):
            payload["temperature"] = self._opts.temperature

        headers = {
            "api-subscription-key": self._opts.api_key,
            "Content-Type": "application/json",
            "User-Agent": sarvam_tts.USER_AGENT,
        }

        try:
            async with self._tts._ensure_session().post(
                url=self._opts.base_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self._conn_options.timeout),
            ) as res:
                if res.status != 200:
                    error_text = await res.text()
                    raise APIStatusError(
                        message=f"Sarvam TTS API Error ({res.status}): {error_text}",
                        status_code=res.status,
                        body=error_text,
                    )

                response_json = await res.json()
                request_id = response_json.get("request_id", "")
                audios = response_json.get("audios", [])
                if not audios or not isinstance(audios, list):
                    raise APIConnectionError("Sarvam TTS API response invalid: no audio data")

                output_emitter.initialize(
                    request_id=request_id or "unknown",
                    sample_rate=self._tts.sample_rate,
                    num_channels=self._tts.num_channels,
                    mime_type="audio/mpeg",
                )
                for audio_b64 in audios:
                    output_emitter.push(base64.b64decode(audio_b64))
        except asyncio.TimeoutError as e:
            raise APITimeoutError("Sarvam TTS API request timed out") from e
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"Sarvam TTS API connection error: {e}") from e


class SarvamMpegTTS(sarvam_tts.TTS):
    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions | None = None,
    ) -> SarvamMpegChunkedStream:
        return SarvamMpegChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options or DEFAULT_API_CONNECT_OPTIONS,
        )
