import os
import pathlib

class Config:

    CWD : str = os.getcwd()
    AUDIO_FOLDER_WD : str = os.path.join(CWD, pathlib.Path("AudioFolder"))
    SAVED_CHECKPOINT_PATH = os.path.join(CWD, "utils", "saved_model", "checkpoint.pth")
    SAVED_W2V2_PATH = os.path.join(CWD, "utils", "saved_model", "w2v2_architecture")

    SAMPLE_RATE = 16000
    LABEL_COLUMNS = ['Repetition']

    NUM_WORKERS = 5
    PIN_MEMORY = False