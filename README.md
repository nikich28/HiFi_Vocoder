### Vocoder using HiFi-GAN

implemented HiFi-GAN, all logs and report are available in wandb

Commands for training and testing:
~~~
!wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
!tar -xjf LJSpeech-1.1.tar.bz2

!git clone https://github.com/nikich28/HiFi_Vocoder.git

%cd HiFi_Vocoder

!pip install -r ./requirements.txt
!pip install torch==1.10.0+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html


#for training
!python3 train.py


#or simply load checkpoint from gdrive
!FILEID='1vcvxjZ5ZFX-OsmmZJsFzJw34qoeYck5D' && \
FILENAME='best_model.pth' && \
FILEDEST="https://docs.google.com/uc?export=download&id=${FILEID}" && \
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate ${FILEDEST} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && rm -rf /tmp/cookies.txt

#for testing
!python3 test.py