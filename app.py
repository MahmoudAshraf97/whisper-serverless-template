from whisper import load_model
import whisperx
import torch
import os
import wget
import json
from omegaconf import OmegaConf

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global whisper_model, alignment_model, metadata
    
    device = 'cuda' if torch.cuda.is_available() else -1
    # Large models result in considerably better and more aligned (words, timestamps) mapping. 
    whisper_model = load_model("large-v2")
    alignment_model, metadata = whisperx.load_align_model(language_code='en',
                                                      device=device,
                                                      model_name = 'WAV2VEC2_ASR_LARGE_LV60K_960H')
    

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global whisper_model, alignment_model, metadata

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    result = model(prompt)

    # Return the results as a dictionary
    return result
