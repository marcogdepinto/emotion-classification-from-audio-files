# Emotion-Classification-Ravdess

# The project

The scope of this project is to create a classifier to predict the emotions of the speaker starting from an audio file.

**Dataset**

For this task, I have used 4948 samples from the RAVDESS dataset (see below to know more about the data).

The samples comes from:

- Audio-only files;

- Video + audio files: I have extracted the audio from each file using the script **Mp4ToWav.py** that you can find in the main directory of the project.

The classes we are trying to predict are the following: (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)

# Result of the application of a Neural Network to this dataset

**95.9%** accuracy on the training set and **91%** on the test set.

# Loss and accuracy plots

![Link to loss](https://github.com/marcogdepinto/Emotion-Classification-Ravdess/blob/master/media/loss.png) 

![Link to accuracy](https://github.com/marcogdepinto/Emotion-Classification-Ravdess/blob/master/media/accuracy.png)

# Tools and languages used

- [Python 3.7](https://www.python.org/downloads/release/python-370/)
- [Google Colab](https://colab.research.google.com/)
- [Google Drive](https://drive.google.com)
- [Jupyter Notebook](http://jupyter.org/)

# About the RAVDESS dataset

**Download**

The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) can be downloaded free of charge at https://zenodo.org/record/1188976. 

**Construction and Validation**

Construction and validation of the RAVDESS is described in our paper: Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391.

The RAVDESS contains 7356 files. Each file was rated 10 times on emotional validity, intensity, and genuineness. Ratings were provided by 247 individuals who were characteristic of untrained adult research participants from North America. A further set of 72 participants provided test-retest data. High levels of emotional validity, interrater reliability, and test-retest intrarater reliability were reported. Validation data is open-access, and can be downloaded along with our paper from PLOS ONE.

**Description**

The dataset contains the complete set of 7356 RAVDESS files (total size: 24.8 GB). Each of the 24 actors consists of three modality formats: Audio-only (16bit, 48kHz .wav), Audio-Video (720p H.264, AAC 48kHz, .mp4), and Video-only (no sound).  Note, there are no song files for Actor_18.

**License information**

“The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)” by Livingstone & Russo is licensed under CC BY-NA-SC 4.0.

**File naming convention**

Each of the 7356 RAVDESS files has a unique filename. The filename consists of a 7-part numerical identifier (e.g., 02-01-06-01-02-01-12.mp4). These identifiers define the stimulus characteristics:

Filename identifiers 

- Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
- Vocal channel (01 = speech, 02 = song).
- Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
- Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the ‘neutral’ emotion.
- Statement (01 = “Kids are talking by the door”, 02 = “Dogs are sitting by the door”).
- Repetition (01 = 1st repetition, 02 = 2nd repetition).
- Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

**Filename example: 02-01-06-01-02-01-12.mp4**

- Video-only (02)
- Speech (01)
- Fearful (06)
- Normal intensity (01)
- Statement “dogs” (02)
- 1st Repetition (01)
- 12th Actor (12)
- Female, as the actor ID number is even.
