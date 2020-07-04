# Audio Emotion Classification from Multiple Datasets

**Executive Summary**

This project presents a deep learning classifier able to predict the emotions of a human speaker encoded in an audio file. The classifier is trained using 2 different datasets, RAVDESS and TESS, and has an overall F1 score of 80% on 8 classes (neutral, calm, happy, sad, angry, fearful, disgust and surprised).

**Feature set information**

For this task, the dataset is built using 5252 samples from:

- the [Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) dataset](https://zenodo.org/record/1188976#.XsAXemgzaUk) 
- the [Toronto emotional speech set (TESS) dataset](https://tspace.library.utoronto.ca/handle/1807/24487) 

The samples include: 

- 1440 speech files and 1012 Song files from **RAVDESS**. This dataset includes recordings of 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions, and song contains calm, happy, sad, angry, and fearful emotions. Each file was rated 10 times on emotional validity, intensity, and genuineness. Ratings were provided by 247 individuals who were characteristic of untrained adult research participants from North America. A further set of 72 participants provided test-retest data. High levels of emotional validity, interrater reliability, and test-retest intrarater reliability were reported. Validation data is open-access, and can be downloaded along with our paper from [PLoS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391).

- 2800 files from **TESS**. A set of 200 target words were spoken in the carrier phrase "Say the word _____' by two actresses (aged 26 and 64 years) and recordings were made of the set portraying each of seven emotions (anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral). There are 2800 stimuli in total. Two actresses were recruited from the Toronto area. Both actresses speak English as their first language, are university educated, and have musical training. Audiometric testing indicated that both actresses have thresholds within the normal range.

The classes the model wants to predict are the following: (0 = neutral, 1 = calm, 2 = happy, 3 = sad, 4 = angry, 5 = fearful, 6 = disgust, 7 = surprised). This dataset is skewed as there is not a calm class in TESS, hence there are less data for that particular class and this is evident when observing the classification report.

Please note that previous versions of this work was developed using only the RAVDESS dataset and TESS has been added recently. Also, the previous versions of this work used audio features extracted from the videos of the RAVDESS dataset. This particular part of the pipeline has been removed because it was shuffling very similar files in the training and test sets, boosting accuracy of the model as a consequence (overfitting). Take a look at [this issue](https://github.com/marcogdepinto/Emotion-Classification-Ravdess/issues/11) to understand more. The old data exploration codebase, including the above mentioned pipeline, is stored in the ```legacy_code``` folder.

**Metrics**

*Model summary*

![Link to model](https://github.com/marcogdepinto/Emotion-Classification-Ravdess/blob/master/media/model.png) 

*Loss and accuracy plots*

![Link to loss](https://github.com/marcogdepinto/Emotion-Classification-Ravdess/blob/master/media/loss.png) 

![Link to accuracy](https://github.com/marcogdepinto/Emotion-Classification-Ravdess/blob/master/media/accuracy.png)

*Classification report*

![Link do classification report](https://github.com/marcogdepinto/Emotion-Classification-Ravdess/blob/master/media/ClassificationReport.png)

*Confusion matrix*

![Link do classification report](https://github.com/marcogdepinto/Emotion-Classification-Ravdess/blob/master/media/ConfusionMatrix.png)

**How to use the code inside this repository**

1)  ```git clone https://github.com/marcogdepinto/Emotion-Classification-Ravdess.git ``` OR, as an alternative, all the optional steps below.

2) *OPTIONAL*: Download Audio_Song_Actors_01-24.zip and Audio_Speech_Actors_01-24.zip, unzip and merge the content of the folders (e.g. Actor_01 should include both Speech and Song) and then add it to the ```features``` folder.

2) *OPTIONAL*: Create two empty folders, ```Actor_25``` and ```Actor_26```, into the ```features``` folder.

3) *OPTIONAL*: Download the TESS dataset and unzip it into the ```TESS_Toronto_emotional_speech_set_data``` folder.
The format you need to have to make the following steps work is:

    ```
    TESS_Toronto_emotional_speech_set_data
    --OAF_angry
    --OAF_disgust
    --Other Folders..
    ```
4) *OPTIONAL*: Run ```tess_pipeline.py```: this will copy the files in the ```Actor_25``` and ```Actor_26``` folders with a usable naming convention. For details, read the docstrings of ```tess_pipeline.py```.

6) *ONLY IF YOU WANT TO CREATE NEW FEATURES*: run ```create_features.py```. Please note this is NOT necessary as in the ```features``` folder there are already the joblib files created with ```create_features.py```.

7) *ONLY IF YOU WANT TO CREATE A NEW MODEL*:  run ```neural_network.py```. Please note this is NOT necessary as in the ```model``` folder there is already a pre_trained model to use.

**How to test the model created in this work**

Let's be clear. When we talk about emotions understanding, we are talking about a very difficult task. 

I have pasted two files in the ```examples``` folder:

a) 03-01-01-01-01-02-05.wav is an example of WRONG prediction: it is a NEUTRAL file, the model predicts CALM. Try to listen to the audio yourself. Which is the emotion for you? For me CALM seems a fair prediction. That speaker is classified as neutral, but he is not angry at all. You see my point?

b) 10-16-07-29-82-30-63.wav is a DISGUST file. The model is getting it.

Feel free to try with other files or record your voice. I still have to try this last one but I am very curious about the result.

*Important note*: the classes are encoded from 0 to 7 in the code. In the dataset, from 01 to 08. Be aware when you try. If the model predicts 0 and you are using a NEUTRAL file (01), this is correct and the expected behavior. Keras wants the predictions to start from 0 and not from 1, so the code is adjusted to cope with this requirement.

**APPENDIX 1: The RAVDESS dataset**

*Download*

The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) can be downloaded free of charge at https://zenodo.org/record/1188976. 

*Construction and Validation*

Construction and validation of the RAVDESS is described in our paper: Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391.

The RAVDESS contains 7356 files. Each file was rated 10 times on emotional validity, intensity, and genuineness. Ratings were provided by 247 individuals who were characteristic of untrained adult research participants from North America. A further set of 72 participants provided test-retest data. High levels of emotional validity, interrater reliability, and test-retest intrarater reliability were reported. Validation data is open-access, and can be downloaded along with our paper from PLOS ONE.

**Description**

The dataset contains the complete set of 7356 RAVDESS files (total size: 24.8 GB). Each of the 24 actors consists of three modality formats: Audio-only (16bit, 48kHz .wav), Audio-Video (720p H.264, AAC 48kHz, .mp4), and Video-only (no sound).  Note, there are no song files for Actor_18.

**License information**

“The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)” by Livingstone & Russo is licensed under CC BY-NA-SC 4.0.

*File naming convention*

Each of the 7356 RAVDESS files has a unique filename. The filename consists of a 7-part numerical identifier (e.g., 02-01-06-01-02-01-12.mp4). These identifiers define the stimulus characteristics:

Filename identifiers 

- Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
- Vocal channel (01 = speech, 02 = song).
- Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
- Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the ‘neutral’ emotion.
- Statement (01 = “Kids are talking by the door”, 02 = “Dogs are sitting by the door”).
- Repetition (01 = 1st repetition, 02 = 2nd repetition).
- Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

*Filename example: 02-01-06-01-02-01-12.mp4*

- Video-only (02)
- Speech (01)
- Fearful (06)
- Normal intensity (01)
- Statement “dogs” (02)
- 1st Repetition (01)
- 12th Actor (12)
- Female, as the actor ID number is even.

**APPENDIX 2: The TESS dataset**

Pichora-Fuller, M. Kathleen; Dupuis, Kate, 2020, "Toronto emotional speech set (TESS)", https://doi.org/10.5683/SP2/E8H2MF, Scholars Portal Dataverse, V1

```
@data{SP2/E8H2MF_2020,
author = {Pichora-Fuller, M. Kathleen and Dupuis, Kate},
publisher = {Scholars Portal Dataverse},
title = "{Toronto emotional speech set (TESS)}",
year = {2020},
version = {DRAFT VERSION},
doi = {10.5683/SP2/E8H2MF},
url = {https://doi.org/10.5683/SP2/E8H2MF}
}
```

**APPENDIX 3: Cite this work**

The paper referred to below uses only the RAVDESS dataset. On the other hand, this github repository includes an updated version of the model that uses the TESS dataset and a different model architecture.

```
@INPROCEEDINGS{9122698,
author={M. G. {de Pinto} and M. {Polignano} and P. {Lops} and G. {Semeraro}},
booktitle={2020 IEEE Conference on Evolving and Adaptive Intelligent Systems (EAIS)},
title={Emotions Understanding Model from Spoken Language using Deep Neural Networks and Mel-Frequency Cepstral Coefficients},
year={2020},
volume={},
number={},
pages={1-5},}
```
