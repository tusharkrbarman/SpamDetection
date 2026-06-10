"""Compatibility wrapper for Sarvam TTS audio responses.

The LiveKit Sarvam plugin currently labels Sarvam's returned bytes as WAV, but
Sarvam can return compressed audio bytes. This wrapper keeps Sarvam as the
provider and converts those bytes to raw PCM before handing them to LiveKit.
"""

from __future__ import annotations

import asyncio
import base64
import io

import aiohttp
import av
from av.audio.resampler import AudioResampler
from livekit.agents import tts
from livekit.agents._exceptions import APIConnectionError, APIStatusError, APITimeoutError
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.plugins.sarvam import tts as sarvam_tts


def _as_list(frame_or_frames: object) -> list[av.AudioFrame]:
    if frame_or_frames is None:
        return []
    if isinstance(frame_or_frames, list):
        return frame_or_frames
    return [frame_or_frames]


def _decode_to_pcm(audio_bytes: bytes, *, sample_rate: int, num_channels: int) -> bytes:
    if not audio_bytes:
        raise ValueError("empty audio payload")

    layout = "mono" if num_channels == 1 else "stereo"
    last_error: Exception | None = None

    for input_format in (None, "mp3", "wav"):
        try:
            with av.open(io.BytesIO(audio_bytes), format=input_format) as container:
                audio_stream = next(
                    (stream for stream in container.streams if stream.type == "audio"),
                    None,
                )
                if audio_stream is None:
                    raise ValueError("payload contains no audio stream")

                resampler = AudioResampler(format="s16", layout=layout, rate=sample_rate)
                chunks: list[bytes] = []
                for packet in container.demux(audio_stream):
                    for frame in packet.decode():
                        for resampled in _as_list(resampler.resample(frame)):
                            chunks.append(resampled.to_ndarray().tobytes())

                for resampled in _as_list(resampler.resample(None)):
                    chunks.append(resampled.to_ndarray().tobytes())

                pcm = b"".join(chunks)
                if not pcm:
                    raise ValueError("payload decoded without audio frames")
                return pcm
        except Exception as e:  # noqa: BLE001 - try the next plausible container.
            last_error = e

    raise ValueError(f"unable to decode Sarvam audio payload: {last_error}") from last_error


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

                compressed_audio = b"".join(base64.b64decode(audio_b64) for audio_b64 in audios)
                try:
                    pcm_audio = _decode_to_pcm(
                        compressed_audio,
                        sample_rate=self._tts.sample_rate,
                        num_channels=self._tts.num_channels,
                    )
                except ValueError as e:
                    audio_prefix = compressed_audio[:8].hex()
                    raise APIConnectionError(
                        "Sarvam TTS response could not be decoded "
                        f"(bytes={len(compressed_audio)}, prefix={audio_prefix})"
                    ) from e

                output_emitter.initialize(
                    request_id=request_id or "unknown",
                    sample_rate=self._tts.sample_rate,
                    num_channels=self._tts.num_channels,
                    mime_type="audio/pcm",
                )
                output_emitter.push(pcm_audio)
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
