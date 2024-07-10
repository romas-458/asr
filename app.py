import numpy as np
import librosa
import torch

from math import ceil
import nemo.collections.asr as nemo_asr

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import torch
import tempfile
import soundfile as sf


asr_model = nemo_asr.models.EncDecCTCModelBPE. \
                    from_pretrained("theodotus/stt_uk_squeezeformer_ctc_ml",map_location="cpu")

asr_model.preprocessor.featurizer.dither = 0.0
asr_model.preprocessor.featurizer.pad_to = 0
asr_model.eval()
asr_model.encoder.freeze()
asr_model.decoder.freeze()


buffer_len = 8.0
chunk_len = 4.8
total_buffer = round(buffer_len * asr_model.cfg.sample_rate)
overhead_len = round((buffer_len - chunk_len) *  asr_model.cfg.sample_rate)
model_stride = 4


model_stride_in_secs = asr_model.cfg.preprocessor.window_stride * model_stride
tokens_per_chunk = ceil(chunk_len / model_stride_in_secs)
mid_delay = ceil((chunk_len + (buffer_len - chunk_len) / 2) / model_stride_in_secs)


def resample(audio):
    audio_16k, sr = librosa.load(audio, sr = asr_model.cfg["sample_rate"], 
                            mono=True,  res_type='soxr_hq')
    return audio_16k


def model(audio_16k):
    logits, logits_len, greedy_predictions = asr_model.forward(
        input_signal=torch.tensor([audio_16k]), 
        input_signal_length=torch.tensor([len(audio_16k)])
    )
    return logits


def decode_predictions(logits_list):
    logits_len = logits_list[0].shape[1]
    # cut overhead
    cutted_logits = []
    for idx in range(len(logits_list)):
        start_cut = 0 if (idx==0) else logits_len - 1 - mid_delay
        end_cut = -1 if (idx==len(logits_list)-1) else logits_len - 1 - mid_delay + tokens_per_chunk
        logits = logits_list[idx][:, start_cut:end_cut]
        cutted_logits.append(logits)

    # join
    logits = torch.cat(cutted_logits, axis=1)
    logits_len = torch.tensor([logits.shape[1]])
    current_hypotheses, all_hyp = asr_model.decoding.ctc_decoder_predictions_tensor(
        logits, decoder_lengths=logits_len, return_hypotheses=False,
    )

    return current_hypotheses[0]


def transcribe_func(audio):
    state = [np.array([], dtype=np.float32), []]

    audio_16k = resample(audio)

    # join to audio sequence
    state[0] = np.concatenate([state[0], audio_16k])

    while (len(state[0]) > overhead_len) or (len(state[1]) == 0):
        buffer = state[0][:total_buffer]
        state[0] = state[0][total_buffer - overhead_len:]
        # run model
        logits = model(buffer)
        # add logits
        state[1].append(logits)

    if len(state[1]) == 0:
        text = ""
    else:
        text = decode_predictions(state[1])
    return text


app = FastAPI()

class TranscriptionResult(BaseModel):
    transcription: str


@app.post("/transcribe/", response_model=TranscriptionResult)
async def transcribe(file: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.file.read())
        temp_file_path = temp_file.name

    # Perform transcription
    transcription = transcribe_func(temp_file_path)

    return TranscriptionResult(transcription=transcription)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

    # y, sr = librosa.load('common_voice_uk_38506506.mp3')

    # y, sr = librosa.load('common_voice_uk_38506506.mp3', sr=asr_model.cfg["sample_rate"], mono=True, res_type='soxr_hq')
    # print(y)