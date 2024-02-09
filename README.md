# Disfluency Detection From Text

This repo is related to https://github.com/amritkromana/disfluency_detection_from_audio but requires an input transcript as opposed to an input audio file. 
This repo includes a demo for running text through a BERT-based disfluency detection model. This model was trained with Switchboard. 
It will output predictions for each token in the input file (filled pauses, repetitions, revisions, and restarts). 

## Dependencies 

The following packages are needed: 
- pandas==1.5.0
- torch==1.12.1
- transformers==4.22.2
- gdown==5.1.0

Use gdown to download the pretrained model weights and save to demo_models: 
```
gdown --id 1GQIXgCSF3Usiuy5hkxgOl483RPX3f_SX -O language.pt
```

Alternatively, you can visit this link to download language.pt and save it in the same directory as the code: https://drive.google.com/drive/folders/1O34ut7-U4fE6Ei2ihkqerXfAi1afYC9B

## How to run the demo 

Given some test.txt which contains one line of lowercase text, we can run:  
```
python3 demo.py --input_file test.txt --output_file test.csv
```

# Citation 
This work is a subset of work submitted to IEEE Transactions on Audio, Speech and Language Processing. If you use this work in your research or projects, please cite it as follows:
```
@article{romana2023,
title = {Automatic Disfluency Detection from Untranscribed Speech},
author = {Amrit Romana, Kazuhito Koishida, Emily Mower Provost},
year = {2023}
}
```
