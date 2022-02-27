# TransferLearningAudio
ML.NET porting of https://www.tensorflow.org/tutorials/audio/transfer_learning_audio

```
Audio folder: ..\..\..\..\assets\ESC-50-master

Read model
##########
Model location: ..\..\..\..\assets\yamnet.onnx

Training the ML.NET classification model
########################################
Training with transfer learning took: 6 seconds

Create Predictions and Evaluate the model quality
#################################################

*** Showing all the predictions ***
###################################
ImagePath: 1-30226-A-0.wav original labeled as dog predicted as cat with score 0.9375236
ImagePath: 1-59513-A-0.wav original labeled as dog predicted as cat with score 0.7190483
ImagePath: 1-79113-A-5.wav original labeled as cat predicted as cat with score 0.9999161
ImagePath: 2-82274-A-5.wav original labeled as cat predicted as cat with score 0.99879956
ImagePath: 3-144028-A-0.wav original labeled as dog predicted as dog with score 0.9999914
ImagePath: 3-146964-A-5.wav original labeled as cat predicted as dog with score 0.6153933
ImagePath: 3-146965-A-5.wav original labeled as cat predicted as cat with score 0.51984257
ImagePath: 3-155312-A-0.wav original labeled as dog predicted as dog with score 0.9999728
ImagePath: 3-95698-A-5.wav original labeled as cat predicted as dog with score 0.5606108
ImagePath: 4-133047-B-5.wav original labeled as cat predicted as cat with score 0.84472704
ImagePath: 4-149940-B-5.wav original labeled as cat predicted as cat with score 0.9999995
ImagePath: 4-161303-A-5.wav original labeled as cat predicted as dog with score 0.5856288
ImagePath: 4-191687-A-0.wav original labeled as dog predicted as dog with score 0.99999857
ImagePath: 4-192236-A-0.wav original labeled as dog predicted as dog with score 0.60927355
ImagePath: 4-194754-A-0.wav original labeled as dog predicted as dog with score 0.99703705
ImagePath: 5-203128-B-0.wav original labeled as dog predicted as dog with score 0.9989277
ImagePath: 5-208030-A-0.wav original labeled as dog predicted as dog with score 0.99375653
ImagePath: 5-214759-A-5.wav original labeled as cat predicted as cat with score 0.9999809
ImagePath: 5-214759-B-5.wav original labeled as cat predicted as cat with score 0.997896
ImagePath: 5-259169-A-5.wav original labeled as cat predicted as cat with score 0.8920553
```
