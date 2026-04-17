from speechbrain.inference.classifiers import EncoderClassifier
from collections import Counter
import os
import torch

_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            savedir="pretrained_models/lang-id"
        )
    return _classifier

def detect_language(audio_path: str) -> str:
    classifier = get_classifier()
    prediction = classifier.classify_file(audio_path)
    language = prediction[3][0]
    confidence_tensor = prediction[1]
    if torch.is_tensor(confidence_tensor):
        confidence = float(confidence_tensor.exp().max().item())
    else:
        confidence = float(confidence_tensor)
    return {
        "language":language,
        "confidence": round(confidence, 4)
    }  

def detect_language_for_chunks(chunks_folder: str) -> dict:
    results = {}
    
    if not os.path.exists(chunks_folder):
        return results
    
    for file in sorted(os.listdir(chunks_folder)):
        if file.endswith(".wav"):
            path = os.path.join(chunks_folder, file)
            lang = detect_language(path)
            results[file] = lang
    
    return results


def get_language_stats(results: dict) -> dict:
    if not results:
        return {}
    
    languages = []
    for v in results.values():
        if isinstance(v, dict):
            languages.append(v.get("language", "unknown"))
        else:
            languages.append(str(v))
    
    return dict(Counter(languages))



# import whisper
# from collections import Counter
# import os

# _model = None


# def get_model():
#     global _model
#     if _model is None:
#         _model = whisper.load_model("tiny")  
#     return _model


# def detect_language(audio_path: str) -> str:
#     model = get_model()
#     audio = whisper.load_audio(audio_path)
#     audio = whisper.pad_or_trim(audio)       
#     mel = whisper.log_mel_spectrogram(audio).to(model.device)
#     _, probs = model.detect_language(mel)
#     language = max(probs, key=probs.get)
#     confidence = round(probs[code], 4)   
#     return {
#         "language":language,
#         "confidence": confidence
#         }  


# def detect_language_for_chunks(chunks_folder: str) -> dict:
   
#     results = {}

#     if not os.path.exists(chunks_folder):
#         return results

#     for file in sorted(os.listdir(chunks_folder)):
#         if file.endswith(".wav"):
#             path = os.path.join(chunks_folder, file)
#             lang = detect_language(path)
#             results[file] = lang

#     return results


# def get_language_stats(results: dict) -> dict:
#     if not results:
#         return {}

#     languages = []
#     for v in results.values():
#         if isinstance(v, dict):
#             languages.append(v["language"])
#         else:
#             languages.append(v)

#     return dict(Counter(languages))