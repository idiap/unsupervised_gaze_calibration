# Unsupervised Gaze Calibration

This repository is the implementation of the following paper:

    Title: Robust Unsupervised Gaze Calibration using Conversation and Manipulation Attention Priors
    Authors: Rémy Siegfried and Jean-Marc odobez
    Journal: ACM Transactions on Multimedia Computing, Communications, and Applications
    Year: 2021

Please cite the above work when using resources from this repo.
Contact: remy.siegfried@idiap.ch, odobez@idiap.ch


# 1. Data

To dataset used for the above publication is available her: https://www.idiap.ch/en/dataset/gaze_vfoa_conversation_and_manipulation

The code is meant to be used on data stored in txt (one video frame by line) with the following fields separated by ",":
* id: identifier for the subject/session (usually something like "<session>_<subject_role>" )
* frameIndex: number of the frame (first video frame has index 0)
* eye (x, y, z): 3D position of the eye in the HCS frame (see below)
* gaze (x, y, z): 3D gaze vector in the HCS frame (see below), i.e. the line of sight with origin on the eye
* head (x, y, z): 3D position of the head (i.e. nose tip) in the CCS frame (see below)
* headpose (roll, pitch, yaw): orientation of the head in the CCS frame (see below)
* speaking: (int) does the subject speaks ("1") or not ("0"). "-1" stands for no data
* action: (str) action performed by the subject at a given location with the format "<action>:<location>" (see below)
* vfoa_GT: (str) name of the target focused by the subject ("no_data" or "na" are absence of annotation, "blinking" or "blink" are when the subject is blinking, and "aversion" is when the subject does not look at a target; otherwise, <vfoa_GT> is the name of the target the subject is looking at)

Moreover, for each target in the scene:
* name: (str) name of the target
* (x, y, z): 3D position of the target in the HCS frame (see below)
* speaking: (bool) does the target speaks or not (always "0" for objects)

For example, data for a dyadic interaction will have 22 number/str per line (1 target).

## 1.1. Coordinate Systems
We use two different coordinate systems:
* Head Coordinate System (HCS): this frame is attached to the subject's head, more precisely to the nose tip.
** x-axis: to the left from the subject point of view
** y-axis: upward from the subject point of view
** z-axis: in the direction of the head

* Camera Coordinate System (CCS)
** x-axis: to the right from the camera point of view
** y-axis: upward from the camera point of view
** z-axis: backward from the camera point of view

Note that when the subject is in the center of the image and look toward the camera, both coordinate sytems are aligned (thus headpose is (0, 0, 0)).
For computation sake, eye position, target position, and gaze direction are given in the HCS coordinate system.
Note that it is possible to transform them into CCS using the head and headpose information.


## 1.2. Actions
We consider two actions:
* grasp: indicates when the subject touch the object before a pick and place
* release: indicated when the subject release the object after pick and place

Also, the location where the action takes place (i.e. position of the manipulated object) is given as the name of the corresponding target, representing the object position in term of marker on the table


## 1.3. Data structure and using other data
The current software use a dictonary data structure, which is loaded by the "load_data()" function from "src/data_loaders.py".
It has the following architecture (words in <brackets> design variable that will have another name in the implemented dict):
data = {
    'subject': {
        'identifier': (N, 1) array, (str), name of the subject
        'frameIndex': (N, 1) array, (int), video frame index
        'eye': (N, 3) array, (float), 3D eye position
        'gaze': (N, 3) array, (float), 3D gaze vector
        'head': (N, 3) array, (float), 3D head position
        'headpose': (N, 3) array, (float), 3D head pose (i.e. roll, pitch, yaw)
        'speaking': (N, 1) array, (int), speaking status (0: not speaking, 1; speaking)
        'action': (N, 1) array, (str), action and location with format '<action>:<location>'
        'vfoa_gt': (N, 1) array, (str), name of the VFOA target
    },
    'targets': {
    	'<target_name>': (N, 4) array, (float), 3D target position, followed by speaking status (0 or 1)
    }
}
Note that data['targets'] can have any number of target in it.

To use other data or in case of missing data, several fields (namely identifier, frameIndex, and head)
can be filled with nan values without impact.



# 2. Scripts

The "src" folder contains the main scripts to perform calibration.
The "offline_calibration.py" and "online_calibration.py" script allows to perform calibration end to end, taking pieces of code from "src" folder together.
The "paper_experiments" folder contains scripts to perform the same experiments as in the related paper.


## 2.1. Package requirements

This code was tested on python2.7.
Required packages are listed in requirements.txt.


## 2.2. Example

Let's say that data txt files are in "dataset/data",
a quick example can be launched using:

>>>python offline_calibration.py -calib dataset/data/data_UBImpressed_025BM_Interviewer.txt -eval dataset/data/data_UBImpressed_025BM_Interviewer.txt -config config/speaking_constant.json -out results_offline_ubimpressed.txt -v

# License

This code is licensed under GPL.
