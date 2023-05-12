"""
Configuration file: using this one to keep all the paths in one place for various imports.

TRAINING_FILES_PATH = Path of the training files. Here there are

- the RAVDESS dataset files (Folders Actor_01 to Actor_24
- the TESS dataset renamed files (Folders Actor_25 and Actor_26)

SAVE_DIR_PATH = Path of the joblib features created with create_features.py

MODEL_DIR_PATH = Path for the keras model created with neural_network.py

TESS_ORIGINAL_FOLDER_PATH = Path for the TESS dataset original folder (used by tess_pipeline.py)

"""
import pathlib
import os

working_dir_path = pathlib.Path().absolute()

TRAINING_FILES_PATH = os.path.join(str(working_dir_path), 'features')
SAVE_DIR_PATH = os.path.join(str(working_dir_path), 'joblib_features')
MODEL_DIR_PATH = os.path.join(str(working_dir_path), 'model')
TESS_ORIGINAL_FOLDER_PATH = os.path.join(str(working_dir_path), 'TESS_Toronto_emotional_speech_set_data')
EXAMPLES_PATH = os.path.join(str(working_dir_path), 'examples')
