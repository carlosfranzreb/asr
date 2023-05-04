import os
from argparse import ArgumentParser
import whisper


def transcribe(model, input_path, output_path, lang):
    with open(output_path, "w") as f:
        f.write(model.transcribe(input_path, language=lang)["text"])


def transcribe_dir(input_dir, output_dir, model, lang, ext):
    model = whisper.load_model(model)
    # Iterate over the input directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(ext):
                rel_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                os.makedirs(output_path, exist_ok=True)
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_path, file)
                output_path = output_path[: -len(ext)] + "txt"
                transcribe(model, input_path, output_path, lang)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_folder", help="Folder containing audio files")
    parser.add_argument("--dump_folder", help="Folder to dump text files")
    parser.add_argument("--model", help="Path to the model")
    parser.add_argument("--language", help="Language of the audio files")
    parser.add_argument("--ext", help="Extension of the audio files")
    args = parser.parse_args()
    transcribe_dir(
        args.data_folder, args.dump_folder, args.model, args.language, args.ext
    )
