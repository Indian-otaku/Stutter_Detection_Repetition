from fastapi import FastAPI, File, UploadFile
import uvicorn
import os
import torch

from convert import convert_bytes_to_file
from config import Config
from get_data import get_batched_data
from get_model import get_pretrained_model

app = FastAPI()

accepted_file_types = [
    "audio/wav",
    "audio/mpeg",
    "audio/wave"
]

@app.post("/api/repetition")
async def general_audio_info(audio: UploadFile):

    if audio.content_type not in accepted_file_types:
        return {"error": f"Enter a valid file type. Given type is {audio.content_type}"}
    
    audio_data = await audio.read()
    file_name = convert_bytes_to_file(audio_data, audio.content_type)
    
    if file_name:
        batched_data = get_batched_data(os.path.join(Config.AUDIO_FOLDER_WD, file_name))
        os.remove(os.path.join(Config.AUDIO_FOLDER_WD, file_name))
        model = get_pretrained_model()
        logits = model(batched_data)
        probs = torch.sigmoid(logits).squeeze()
        prediction = torch.round(probs)
        confidence = torch.where((prediction == 1), probs, 1 - probs)

        return {
            'prediction': prediction.tolist(),
            'confidence': confidence.tolist()
        }
        
    
    return {"error": "Invalid file"}
    

if __name__ == '__main__':
	uvicorn.run(app, port=10000, host="0.0.0.0")
