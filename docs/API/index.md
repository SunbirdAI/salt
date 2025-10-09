# Sunbird API — Tasks Endpoints (Developer Reference)

**Version:** 1.0  
**Last updated:** 2025-10-08  
**Maintainer:** Sunbird AI

Welcome to the Sunbird AI API documentation. The Sunbird AI API provides you access to Sunbird's language models. 
## Table of Contents

- [Quickstart](#quickstart)
- [Global behavior and requirements](#global-behavior-and-requirements)
- [POST /tasks/stt — Speech-to-text](#stt)
- [POST /tasks/nllb_translate — Neural translation](#translate)
- [POST /tasks/language_id — Language identification](#languageid)
- [POST /tasks/summarise — Text summarisation](#summarise)
- [POST /tasks/tts — Text-to-Speech (TTS)](#tts)
- [Error handling & retries](#error-handling--retries)
- [Practical tips & checklist](#practical-tips--checklist)
- [Troubleshooting & Support](#troubleshooting--support)

<a name="quickstart"></a>
## Quickstart

1. Get an API key from your Sunbird dashboard.
2. Call any endpoint with Authorization: Bearer YOUR_API_KEY.

Example header (all requests):

```
Authorization: Bearer YOUR_API_TOKEN
Content-Type: application/json
```

If you use multipart uploads (STT), set Content-Type to multipart/form-data in your client.

---

<a name="global-behavior-and-requirements"></a>
## Rate limiting and quotas

Rate limits are enforced by account type (derived from JWT `account_type`):

- **Free Tier:** Limited requests per hour
- **Professional:** Higher rate limits
- **Enterprise:** Custom rate limits

On 429 responses, implement exponential backoff and jitter. Monitor usage to avoid throttling.

---
# Sunbird API — Tasks Endpoints (Authoritative Reference)

Version: 1.3
Last updated: 2025-10-08
Maintainer: Sunbird AI

This document covers exactly the five /tasks endpoints requested:

- POST /tasks/stt
- POST /tasks/nllb_translate
- POST /tasks/language_id
- POST /tasks/summarise
- POST /tasks/tts

Each section below contains: purpose, precise request/response schema, cURL/Python examples, common errors, limits, and integration tips.

---

## Global behavior and requirements

- Authentication: All endpoints require an `Authorization: Bearer <API_KEY>` header.
- Content-Type: `application/json` for JSON endpoints; `multipart/form-data` for file uploads (STT).
- Rate limiting: enforced per account_type (admin: 1000/min, premium: 100/min, standard: 50/min).
- Timeouts: Server-side model/worker timeouts may return 408/503/504. These are retryable in most cases.

Signed storage URLs
- TTS returns a signed Google Cloud Storage URL in `output.audio_url`. These URLs are time-limited (typically ~1800s). If you need persistent access, download and store the file server-side, or use `output.blob` to re-create a signed URL from server-side credentials.

Error response format

```json
{ "detail": "Human readable error message" }
```

---

<a name="stt"></a>
# POST /tasks/stt — Speech-to-text (file upload)

### Purpose
- Upload an audio file and receive a transcription. Optional speaker diarization and whisper-assisted decoding are available.

### Limits & behaviour
- Supported file types: mp3, wav, ogg, m4a, aac.
- Max processed duration: 10 minutes. Longer audio is trimmed to first 10 minutes server-side; `was_audio_trimmed` will be set to `true` in the response.
- For very large files (>100MB) use the signed upload flow and call `/tasks/stt_from_gcs`.

### Request (multipart/form-data)
- audio: file (required)
- language: `SttbLanguage` enum (default: `lug`)
- adapter: `SttbLanguage` enum (default: `lug`)
- recognise_speakers: boolean (default: false)
- whisper: boolean (default: false)

### Examples (cURL / Python)
#### cURL example (multipart)

```bash
curl -X POST "https://api.sunbird.ai/tasks/stt" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "audio=@sample.mp3;type=audio/mpeg" \
  -F "language=lug" \
  -F "adapter=lug" \
  -F "recognise_speakers=false" \
  -F "whisper=false"
```

Python example (requests)

```python
import requests

url = 'https://api.sunbird.ai/tasks/stt'
headers = {'Authorization': 'Bearer YOUR_API_KEY'}
files = {'audio': open('sample.mp3', 'rb')}
data = {'language': 'lug', 'adapter': 'lug'}

resp = requests.post(url, headers=headers, files=files, data=data, timeout=180)
resp.raise_for_status()
print(resp.json())
```

### Successful response (STTTranscript)

```json
{
  "audio_transcription": "Transcribed text...",
  "diarization_output": {},
  "formatted_diarization_output": "Speaker 1: ...\nSpeaker 2: ...",
  "audio_transcription_id": 123,
  "audio_url": "gs://<bucket>/<path>",
  "language": "lug",
  "was_audio_trimmed": false,
  "original_duration_minutes": null
}
```

#### Request Parameters (table)

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| audio | file | Yes | - | Audio file to transcribe (mp3/wav/ogg/m4a/aac) |
| language | string | No | "lug" | Target language / transcription model adapter |
| adapter | string | No | "lug" | Adapter preference for model (same enum as language) |
| recognise_speakers | boolean | No | false | Enable speaker diarization |
| whisper | boolean | No | false | Use whisper-assisted decoding if available |

#### Response Fields (table)

| Name | Type | Description |
|------|------|-------------|
| audio_transcription | string | Full transcription text |
| diarization_output | object | Detailed diarization data (timestamps, speaker ids) |
| formatted_diarization_output | string | Human-readable speaker-labelled transcript |
| audio_transcription_id | integer/null | Optional DB id for stored transcription |
| audio_url | string | GCS path (if stored) |
| language | string | Detected/used language code |
| was_audio_trimmed | boolean | True if original audio exceeded processing limit |
| original_duration_minutes | float/null | Original audio length when trimmed |


### Common errors

Error response examples

- 400 Bad Request — Missing file

```bash
curl -i -X POST "https://api.sunbird.ai/tasks/stt" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

HTTP/1.1 400 Bad Request
```json
{ "detail": "Missing 'audio' file in request" }
```

- 422 Unprocessable Entity — No transcription produced

```bash
curl -i -X POST "https://api.sunbird.ai/tasks/stt" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "audio=@silence.mp3;type=audio/mpeg"
```

HTTP/1.1 422 Unprocessable Entity
```json
{ "detail": "No transcription generated from the provided audio" }
```

- 503 Service Unavailable — Worker timeout

```bash
curl -i -X POST "https://api.sunbird.ai/tasks/stt" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "audio=@large_file.mp3;type=audio/mpeg"
```

HTTP/1.1 503 Service Unavailable
```json
{ "detail": "Transcription worker timed out. Please retry later." }
```
- 400 — Missing file or invalid parameters.
- 415 — Unsupported media type.
- 422 — No transcription produced.
- 503 / 504 — Worker timeout or backend unavailable (retryable).

Integration tips
- Re-encode to 16 kHz, mono, 16-bit PCM to minimize decoding errors.
- If you control the client, chunk long audio into smaller segments and send multiple requests.

---

<a name="translate"></a>
# POST /tasks/nllb_translate — Neural translation

### Purpose
- Translate text between English and supported local languages.

Supported codes: `ach`, `teo`, `eng`, `lug`, `lgg`, `nyn`.

### Request (application/json)

```json
{
  "source_language": "eng",
  "target_language": "lug",
  "text": "How are you?"
}
```

### Examples (cURL)

```bash
curl -X POST "https://api.sunbird.ai/tasks/nllb_translate" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"source_language":"eng","target_language":"lug","text":"How are you?"}'
```

### Response (normalized worker wrapper)

```json
{
  "id": "job-id",
  "status": "success",
  "executionTime": 123,
  "output": {
    "text": "How are you?",
    "translated_text": "Oli otya?",
    "source_language": "eng",
    "target_language": "lug",
    "Error": null
  }
}
```

  Request Parameters (table)

  | Name | Type | Required | Default | Description |
  |------|------|----------|---------|-------------|
  | source_language | string | Yes | - | Source language code (ach|teo|eng|lug|lgg|nyn) |
  | target_language | string | Yes | - | Target language code |
  | text | string | Yes | - | Text to translate |

  Response Fields (table)

  | Name | Type | Description |
  |------|------|-------------|
  | id | string | Job identifier |
  | status | string | Job status (success/failure) |
  | executionTime | integer | Execution time in ms |
  | output.text | string | Original input text (optional) |
  | output.translated_text | string | Translated text |
  | output.source_language | string | Source language returned by worker |
  | output.target_language | string | Target language returned by worker |
  | output.Error | string/null | Error message if translation failed |
  | workerId | string/null | Worker instance id |

### Errors & retries
- 400 — invalid or missing fields.
- 503 — transient worker failures (retry with backoff).

Integration tips
- Keep text short (<512 tokens) for consistent quality.
- For bulk translations, run sequential requests or implement a batching layer.

---

<a name="languageid"></a>
# POST /tasks/language_id — Language identification

### Purpose
- Identify which language (from a limited set) a short text is written in.

### Request

```json
{ "text": "Nkwagala nnyo" }
```

cURL example

```bash
curl -X POST "https://api.sunbird.ai/tasks/language_id" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text":"Nkwagala nnyo"}'
```

### Response

```json
{ "language": "lug" }
```

Request Parameters (table)

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| text | string | Yes | - | Text to identify language for |

Response Fields (table)

| Name | Type | Description |
|------|------|-------------|
| language | string | Detected language code (ach|teo|eng|lug|lgg|nyn) or `language not detected` |
| confidence | float/null | (Optional) Confidence score if provided by the service |

Notes
- Supported output labels: `ach`, `teo`, `eng`, `lug`, `lgg`, `nyn`.
- If confidence is low the service may return `language not detected`.

- Aggregate predictions across sentences for mixed or longer text.
### Errors
Integration tips
- Aggregate predictions across sentences for mixed or longer text.
- Aggregate predictions across sentences for mixed or longer text.

Error response examples

- 400 Bad Request — Missing `text` field

```bash
curl -i -X POST "https://api.sunbird.ai/tasks/language_id" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{}'
```

HTTP/1.1 400 Bad Request
```json
{ "detail": "Missing required field: text" }
```

- 422 Unprocessable Entity — Low confidence / not detected

```bash
curl -i -X POST "https://api.sunbird.ai/tasks/language_id" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text":"..."}'
```

HTTP/1.1 422 Unprocessable Entity
```json
{ "detail": "Language not detected with sufficient confidence" }
```

- 503 Service Unavailable — Worker unavailable

```bash
curl -i -X POST "https://api.sunbird.ai/tasks/language_id" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text":"Nkwagala nnyo"}'
```

HTTP/1.1 503 Service Unavailable
```json
{ "detail": "Language identification service unavailable. Please retry later." }
```

---

<a name="summarise"></a>
# POST /tasks/summarise — Text summarisation

### Purpose
- Produce anonymised short summaries of long text.

### Request

```json
{ "text": "Very long article or conversation..." }
```

cURL example

```bash
curl -X POST "https://api.sunbird.ai/tasks/summarise" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text":"Very long text to summarise..."}'
```

### Response

```json
{ "summarized_text": "Short summary..." }
```

Request Parameters (table)

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| text | string | Yes | - | Text to summarise (longer texts can be chunked) |

Response Fields (table)

| Name | Type | Description |
|------|------|-------------|
| summarized_text | string | Generated short summary |
| language | string/null | (Optional) Detected language of the input |
| processing_time | float/null | (Optional) Server processing time in seconds |

- Supported languages: English and Luganda.
- For very long texts chunk and summarise incrementally, then combine.
- The service applies basic anonymisation heuristics; do not rely on it for legal PII removal.
### Notes
- Supported languages: English and Luganda.
- For very long texts chunk and summarise incrementally, then combine.
- The service applies basic anonymisation heuristics; do not rely on it for legal PII removal.
- Supported languages: English and Luganda.
- For very long texts chunk and summarise incrementally, then combine.
- The service applies basic anonymisation heuristics; do not rely on it for legal PII removal.

---

<a name="tts"></a>
# POST /tasks/tts — Text-to-Speech (TTS)

### Text-to-Speech (TTS) API Documentation

### Overview
- The TTS endpoint converts text to speech using Sunbird AI's multilingual text-to-speech models. This service supports multiple Ugandan languages and provides high-quality voice synthesis for applications such as IVR, accessibility, language learning, and content generation.

Endpoint Details
- URL: POST /tasks/tts
- Authentication: Bearer Token required
- Rate Limiting: Applied based on account type (Admin: 1000/min, Premium: 100/min, Standard: 50/min)

### Supported Languages & Voices

| Language    | Speaker ID | Voice Type | Description                       |
|-------------|------------|------------|-----------------------------------|
| Acholi      | 241        | Female     | Native Acholi speaker             |
| Ateso       | 242        | Female     | Native Ateso speaker              |
| Runyankole  | 243        | Female     | Native Runyankole speaker         |
| Lugbara     | 245        | Female     | Native Lugbara speaker            |
| Swahili     | 246        | Male       | Native Swahili speaker            |
| Luganda     | 248        | Female     | Native Luganda speaker (default)  |

### Request Format

Headers

```
Authorization: Bearer YOUR_API_TOKEN
Content-Type: application/json
```

Request Body (JSON)

```json
{
  "text": "Oli otya? Nkwagala nnyo.",
  "speaker_id": 248,
  "temperature": 0.7,
  "max_new_audio_tokens": 2000
}
```

Parameters

| Parameter            | Type    | Required | Default | Range     | Description                                  |
|----------------------|---------|----------|---------|-----------|----------------------------------------------|
| text                 | string  | Yes      | -       | 1-5000    | Text to convert to speech                    |
| speaker_id           | integer | No       | 248     | See table | Voice/speaker selection                       |
| temperature          | float   | No       | 0.7     | 0.0-2.0   | Voice expression control                      |
| max_new_audio_tokens | integer | No       | 2000    | 100-5000  | Maximum audio length (generation budget)      |

Parameter Details

- text: The input text to be synthesized. Supports Unicode characters for local languages. Keep individual requests under 5,000 characters where possible.
- speaker_id: Selects the voice and language. Each ID corresponds to a specific language and gender (see table above).
- temperature: Controls voice expressiveness:
  - 0.0-0.3: More monotone, consistent pronunciation
  - 0.4-0.7: Balanced, natural expression (recommended)
  - 0.8-2.0: More expressive, variable intonation
- max_new_audio_tokens: Limits the maximum length of generated audio to control processing time and costs. Increase for longer outputs, but keep within sensible bounds to avoid timeouts.

### Response Format

Success Response (200 OK)

```json
{
  "output": {
    "audio_url": "https://storage.googleapis.com/sb-asr-audio-content-sb-gcp-project-01/tts/20251003082338_uuid.mp3?X-Goog-...",
    "duration_seconds": 4.2,
    "blob": "tts/20251003082338_uuid.mp3",
    "sample_rate": 16000,
    "format": "mp3",
    "speaker_id": 248,
    "processing_time": 1.8
  }
}
```

Response Fields

| Field                    | Type    | Description                                                       |
|--------------------------|---------|-------------------------------------------------------------------|
| output.audio_url         | string  | Signed URL to directly download the generated audio file           |
| output.duration_seconds  | float   | Length of the generated audio in seconds                          |
| output.blob              | string  | Cloud storage blob path / identifier for the generated audio file |
| output.sample_rate       | integer | Audio sample rate in Hz                                            |
| output.format            | string  | Audio file format (e.g., "mp3")                                  |
| output.speaker_id        | integer | Confirmation of speaker used                                        |
| output.processing_time   | float   | Server processing time in seconds                                   |

Important notes about the response

1. Signed URL expiry: The `audio_url` is a time-limited signed URL that expires after 2 minutes (120 seconds). Download the file immediately or store the `blob` and generate a new signed URL server-side when needed.
2. File format: The service produces MP3 audio by default; `format` indicates the actual container/codec returned.
3. Sample rate: Typical sample rate is 16000 Hz unless otherwise noted.
4. Blob naming: `blob` includes a timestamp and unique identifier for traceability (e.g., `tts/YYYYMMDD_HHMMSS_uuid.mp3`).


### Error Responses

- 400 Bad Request — validation errors (missing `text`, invalid `speaker_id`, out-of-range `temperature`)
- 503 Service Unavailable — worker timeout or connection issues (retryable with backoff)
- 500 Internal Server Error — unexpected internal errors

Error response examples

- 400 Bad Request — Invalid or missing parameter (example: invalid speaker_id)

```bash
curl -i -X POST "https://api.sunbird.ai/tasks/tts" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text":"Webale","speaker_id":9999}'
```

HTTP/1.1 400 Bad Request
```json
{ "detail": "Invalid speaker_id: 9999" }
```

- 503 Service Unavailable — Worker timeout / transient failure

```bash
curl -i -X POST "https://api.sunbird.ai/tasks/tts" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text":"Long text that may trigger a timeout...","speaker_id":248}'
```

HTTP/1.1 503 Service Unavailable
```json
{ "detail": "TTS worker timed out. Please retry later." }
```

- 500 Internal Server Error — Generation error

```bash
curl -i -X POST "https://api.sunbird.ai/tasks/tts" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text":"Webale nyo","speaker_id":248}'
```

HTTP/1.1 500 Internal Server Error
```json
{ "detail": "Internal server error during audio generation" }
```


### Examples (cURL / Python / Node)
#### Example cURL (request + immediate download)

```bash
# 1) Request TTS and get signed URL
curl -s -X POST "https://api.sunbird.ai/tasks/tts" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text":"Webale nyo","speaker_id":248}' \
  -o tts_response.json

# 2) Extract audio_url and download (shell example assumes jq is installed)
audio_url=$(jq -r '.output.audio_url' tts_response.json)
curl -L "$audio_url" --output output.mp3
```

Client considerations

- Immediate download: Because the signed `audio_url` expires in 2 minutes, clients should download the audio immediately after receiving the TTS response.
- Server-side re-signing: If you need long-term access, call your server with the stored `blob` path and have the server generate a fresh signed URL using service credentials.
- Cost control: Cache `blob` for identical texts to avoid repeated synthesis costs.

Python example — request TTS and download immediately

```python
import requests

API_URL = 'https://api.sunbird.ai/tasks/tts'
HEADERS = {'Authorization': 'Bearer YOUR_API_KEY', 'Content-Type': 'application/json'}

payload = {"text": "Webale nyo", "speaker_id": 248}

# 1) Request TTS
resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
resp.raise_for_status()
data = resp.json()
audio_url = data['output']['audio_url']

# 2) Download signed URL immediately
with requests.get(audio_url, stream=True, timeout=30) as r:
    r.raise_for_status()
    with open('output.mp3', 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

print('Saved output.mp3')
```

Node.js example — request TTS and stream download

```javascript
const axios = require('axios');
const fs = require('fs');

async function ttsAndDownload(apiKey, text, outFile='output.mp3'){
  const url = 'https://api.sunbird.ai/tasks/tts';
  const payload = { text, speaker_id: 248 };

  // 1) Request TTS
  const resp = await axios.post(url, payload, {
    headers: { Authorization: `Bearer ${apiKey}` },
    timeout: 60000,
  });

  const audioUrl = resp.data.output.audio_url;

  // 2) Stream download
  const audioResp = await axios.get(audioUrl, { responseType: 'stream', timeout: 30000 });
  const writer = fs.createWriteStream(outFile);
  audioResp.data.pipe(writer);

  return new Promise((resolve, reject) => {
    writer.on('finish', () => resolve(outFile));
    writer.on('error', reject);
  });
}

// Example usage:
(async () => {
  const apiKey = 'YOUR_API_KEY';
  try {
    const file = await ttsAndDownload(apiKey, 'Webale nyo');
    console.log('Saved', file);
  } catch (err) {
    console.error('Error:', err.message);
  }
})();
```


---

<a name="error-handling--retries"></a>
## Error handling & retries

- 400 — Client validation error. Check required fields and types.
- 401 — Unauthorized. Check API key.
- 422 — Unprocessable (STT failure to produce transcription).
- 429 — Rate limited. Backoff with jitter.
- 503 / 504 — Transient worker/network issues. Retry with exponential backoff.

Suggested retry pattern (pseudo):

```python
import time, random

def retry(fn, attempts=4):
    for i in range(attempts):
        try:
            return fn()
        except (TemporaryFailure,) as e:
            delay = (2 ** i) + random.random()
            time.sleep(delay)
    raise
```

---

<a name="practical-tips--checklist"></a>
## Practical tips & checklist for integration

- Validate inputs client-side (length, encoding, supported language codes).
- For STT, prefer uploading re-encoded audio (16kHz mono) to reduce failures.
- Always check and handle `was_audio_trimmed` for long recordings.
- Download TTS audio immediately or store `blob` to regenerate signed URLs server-side.
- Monitor API usage to avoid rate-limits; implement queuing for bursts.

---

If you want, I can also:

- generate an HTML version of this single-file reference for hosting,
- add additional sample languages for `nllb_translate`, or
- include sample unit tests that exercise each endpoint (mocked).

For support: info@sunbird.ai

---

Copyright © 2025 Sunbird AI. All rights reserved.

- TTS: Cache generated audio by `blob` key. Re-generate signed URL server-side when needed.
- STT: For best results re-encode incoming audio to 16k/16-bit mono before upload if you control the client—this reduces decoding failures.
- Translation: Keep inputs short (< 512 tokens) for consistent quality.

---

<a name="troubleshooting--support"></a>
## Troubleshooting

- 415 Unsupported Media Type — verify client sets correct MIME type and extension.
- 422 No transcription — try re-encoding the audio to mp3/wav.
- 408/504 Timeouts — try splitting input or re-trying with backoff.

If you need further help contact support@sunbird.ai or visit https://api.sunbird.ai/docs.

---

Copyright © 2025 Sunbird AI. All rights reserved.