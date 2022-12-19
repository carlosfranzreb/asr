"""Transcribe audio files in a folder to text files in another folder.
Install whisper with pip install git+https://github.com/openai/whisper.git 
"""


import os
from argparse import ArgumentParser
import whisper


def main(data_folder, dump_folder):
    model = whisper.load_model("large")
    os.makedirs(dump_folder, exist_ok=True)
    # Loop through all the audio files
    for audio_file in os.listdir(data_folder):
        fname, _ = os.path.splitext(audio_file)
        dump_file = os.path.join(dump_folder, fname + ".txt")
        if not os.path.exists(dump_file):
            with open(dump_file, "w") as f:
                # Transcribe the audio file
                f.write(
                    model.transcribe(
                        os.path.join(data_folder, audio_file), language="de"
                    )["text"]
                )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_folder", help="Folder containing audio files")
    parser.add_argument("--dump_folder", help="Folder to dump text files")
    args = parser.parse_args()
    main(args.data_folder, args.dump_folder)
