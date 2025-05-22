**                                                                          Low Light to Ambient Image Converter
**
Steps to train and test the model on your data

Step 1:

Get the dataset in the following format

Data
/High
/Low

Place the ground truth images in High folder, and low light images in Low folder

Do the same for test dataset.

Step 2:

Replace the folder name in data_process.py with your folder name and then convert the dataset to an npz file

Step 3:

Replace the npz file name in lod.py file, and run the file. Both training and test code is present in the file.


Steps to run the run.sh file

You can provide your own test_dataset.npz file and the model file in /Code directory and then run the run.sh file

For convenience I have provided a sample test_dataset.npz file and a  model file (gen_model_009680.h5), which you
can use to run the run.sh file.
