"""
This file builds 2 additional actor folders (25 and 26) using features from the
Toronto emotional speech set (TESS) dataset: https://tspace.library.utoronto.ca/handle/1807/24487

These stimuli were modeled on the Northwestern University Auditory Test No. 6 (NU-6; Tillman & Carhart, 1966).
A set of 200 target words were spoken in the carrier phrase "Say the word _____'
by two actresses (aged 26 and 64 years) and recordings were made of the set portraying each of seven emotions
(anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral). There are 2800 stimuli in total.
Two actresses were recruited from the Toronto area. Both actresses speak English as their first language,
are university educated, and have musical training. Audiometric testing indicated that
both actresses have thresholds within the normal range.

Authors: Kate Dupuis, M. Kathleen Pichora-Fuller

University of Toronto, Psychology Department, 2010.

TESS data can be downloaded from here: https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess/data

To facilitate the feature creation, the TESS data have been renamed using the same naming convention adopted
by the RAVDESS dataset explained below:

Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
Vocal channel (01 = speech, 02 = song).
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the ‘neutral’ emotion.
Statement (01 = “Kids are talking by the door”, 02 = “Dogs are sitting by the door”).
Repetition (01 = 1st repetition, 02 = 2nd repetition).
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

In case of TESS files, an example below. We do not care of assigning values other than the ones
specified below as those are not used by the model, hence we are assigning random integers.
- 03 (Random)
- 01 (Random)
- 01 (This varies according to the fact in TESS we have 1 emotion less then RAVDESS: calm).
- 01 (Random)
- 03 (Random).
- 01 (Random)
- 01 (Random. I thought initially to put 25 if YAF, 26 if OAF, but that is not needed as the pipeline is not
using the actor information from the filename, only the mfccs extracted from librosa and the target emotion).
"""
import os
import shutil
import random

from config import TRAINING_FILES_PATH
from config import TESS_ORIGINAL_FOLDER_PATH


class TESSPipeline:

    @staticmethod
    def create_tess_folders(path):
        """
        We are filling folders Actor_25 if YAF and Actor_26 if OAF.
        The files will be copied and renamed and not simply moved (to avoid messing up
        things during the development of the pipeline.
        Actor_25 and Actor_26 folders must be created before launching this script.
        Example filename: 03-01-07-02-02-01-01.wav
        """
        counter = 0

        label_conversion = {'01': 'neutral',
                            '03': 'happy',
                            '04': 'sad',
                            '05': 'angry',
                            '06': 'fear',
                            '07': 'disgust',
                            '08': 'ps'}

        for subdir, dirs, files in os.walk(path):
            for filename in files:
                if filename.startswith('OAF'):
                    destination_path = TRAINING_FILES_PATH + 'Actor_26\\'
                    old_file_path = os.path.join(os.path.abspath(subdir), filename)

                    # Separate base from extension
                    base, extension = os.path.splitext(filename)

                    for key, value in label_conversion.items():
                        if base.endswith(value):
                            random_list = random.sample(range(10, 99), 7)
                            file_name = '-'.join([str(i) for i in random_list])
                            file_name_with_correct_emotion = file_name[:6] + key + file_name[8:] + extension
                            new_file_path = destination_path + file_name_with_correct_emotion
                            shutil.copy(old_file_path, new_file_path)

                else:
                    destination_path = TRAINING_FILES_PATH + 'Actor_25\\'
                    old_file_path = os.path.join(os.path.abspath(subdir), filename)

                    # Separate base from extension
                    base, extension = os.path.splitext(filename)

                    for key, value in label_conversion.items():
                        if base.endswith(value):
                            random_list = random.sample(range(10, 99), 7)
                            file_name = '-'.join([str(i) for i in random_list])
                            file_name_with_correct_emotion = (file_name[:6] + key + file_name[8:] + extension).strip()
                            new_file_path = destination_path + file_name_with_correct_emotion
                            shutil.copy(old_file_path, new_file_path)


if __name__ == '__main__':
    TESSPipeline.create_tess_folders(TESS_ORIGINAL_FOLDER_PATH)
