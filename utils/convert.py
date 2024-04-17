from config import Config, os
import pathlib

def convert_bytes_to_file(file_byte_obj : bytes, file_type : str) -> pathlib.Path:
    if file_type == "audio/mpeg":
        file_name = pathlib.Path("input_audio.mp3")
    elif file_type == "audio/wav":
        file_name = pathlib.Path("input_audio.wav")
    elif file_type == "audio/wave":
        file_name = pathlib.Path("input_audio.wave")
    else:
        return None
    if not os.path.exists(Config.AUDIO_FOLDER_WD):
        os.mkdir(Config.AUDIO_FOLDER_WD)
    with open(os.path.join(Config.AUDIO_FOLDER_WD, file_name), "bw") as f:
        f.write(file_byte_obj)
    return file_name
    
