# Bengali Speaker Diarization

This repository contains a [Jupyter notebook](bangla-diarization-neural-vad-embedding-clustering.ipynb) that identifies **when each speaker talks** in Bengali audio. It processes a directory of WAV files and writes one diarization result per file to `submission.csv`. The notebook is configured for the Kaggle Bengali Speaker Diarization Challenge dataset.

## How it works

1. Load each WAV file, mix stereo audio to mono, and resample to 16 kHz.
2. Detect speech regions with Silero neural voice activity detection (VAD).
3. Split speech into overlapping windows (1.5 seconds long, with a 0.75-second hop by default).
4. Extract a speaker embedding from each window with SpeechBrain's pretrained ECAPA-TDNN model (`speechbrain/spkrec-ecapa-voxceleb`).
5. Estimate the speaker count using silhouette scores, then group embeddings with agglomerative clustering using cosine distance.
6. Merge labeled windows into speaker turns and export the results as CSV.

The notebook also plots the waveform with detected speech, a 2D PCA view of the embeddings, and the final speaker timeline for a sample file.

## Run the notebook

1. Open [`bangla-diarization-neural-vad-embedding-clustering.ipynb`](bangla-diarization-neural-vad-embedding-clustering.ipynb) in Kaggle or a Jupyter environment. The notebook's install cell installs `soundfile`, `torchaudio`, `speechbrain`, `scikit-learn`, `tqdm`, and `silero-vad`. A local environment also needs the other imported packages: `numpy`, `pandas`, `torch`, and `matplotlib`.
2. Provide the audio files as `*.wav` in one directory. In the **Paths + Device configuration** cell, set `AUDIO_DIR` to that directory and `OUT_PATH` to the desired CSV path. The current `AUDIO_DIR` points to `/kaggle/input/dl-sprint-4-0-bengali-speaker-diarization-challenge/diarization/diarization/test/audio`.
3. Run the cells in order. The first run downloads the Silero and SpeechBrain model weights, so network access is needed unless they are already cached. The notebook uses CUDA when available for speaker embeddings; VAD runs on CPU.
4. Find `submission.csv` in the notebook's working directory, or at the path you set in `OUT_PATH`.

Check that `AUDIO_DIR` contains WAV files before running the sample visualization cells: the embedding debug cell accesses `wav_paths[0]` directly.

## Output format

The CSV has two columns:

| Column | Content |
| --- | --- |
| `filename` | Input WAV filename without the `.wav` extension. |
| `diarization` | JSON-encoded list of speaker turns. |

Each turn has the following shape:

```json
{"start_time": 13.14, "end_time": 15.66, "speaker_id": "SPEAKER_1"}
```

Times are in seconds and rounded to two decimal places. Speaker IDs are assigned in order of first appearance **within each file**; the same ID in different files does not imply the same person. Files with no detected speech produce an empty list (`[]`).

## Main settings

Change the defaults in `diarize_file()` to tune the pipeline:

| Parameter | Default | Purpose |
| --- | ---: | --- |
| `vad_threshold` | `0.5` | Silero speech detection threshold. |
| `padding_ms` | `300` | Context added around detected speech. |
| `min_silence_ms` | `200` | Minimum silence duration used by VAD. |
| `min_speech_ms` | `300` | Minimum speech duration used by VAD. |
| `win_sec` | `1.5` | Embedding window length in seconds. |
| `hop_sec` | `0.75` | Step between consecutive windows in seconds. |
| `max_speakers` | `6` | Upper bound for automatic speaker-count estimation. |

The speaker count is a heuristic, and the notebook does not transcribe speech or identify speakers across recordings. Inspect the plots and predicted turns when assessing the results.
