import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForAudioClassification

from config import Config


class Wav2Vec2Model(nn.Module):
    def __init__(
            self,
            config=Config
    ):
        super(Wav2Vec2Model, self).__init__()
        self.config = config

        if os.path.exists(Config.SAVED_W2V2_PATH):
            self.model = AutoModelForAudioClassification.from_pretrained(Config.SAVED_W2V2_PATH)
        else:
            self.model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base", num_labels=1)
            self.model.save_pretrained(Config.SAVED_W2V2_PATH)

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_data):
        out = self.model(input_data).logits
        return out
    
def get_pretrained_model():
    model = Wav2Vec2Model()
    checkpoint = torch.load(Config.SAVED_CHECKPOINT_PATH)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model