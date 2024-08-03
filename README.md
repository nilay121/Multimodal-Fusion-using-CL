# Multimodal-Fusion-using-CL
Fusing the information from two different modalities of data for real time object classfication using non-exemplar continual learning algorithm "".

## Install the dependencies in a virtual environment

- Create a virtual environment (Python version 3.8.10) 
  
  ```bash
  python3 -m venv Multimodal_cl
  ```

- Activate the virtual environment
  ```bash
  . Multimodal_cl/bin/activate
  
- Install the dependencies

  ```bash
  pip3 install -r requirements.txt
  ```
## Steps to follow
- Put the gripper dataset in the "dataset" folder
- Run the "dataset_vidToImage.py" file to extract train test images from video frame
- Run the "unsupervised_dataset.py" file to generate the unlabeled data for SSL
- Put the pre-trained feature extractors in the "pre_trained_models" folder

## Ros dependencies
- Install ROS1 (Noetic Ninjemys distribution) on Ubuntu 20.04.
- Follow the steps provided in the "ros_instruction" file to create the ROS package.
- Copy paste the python scripts for publisher and subscriber nodes alongwith the pre-trained feature extractor and the saved matrices to the dedicated folders.

## Different combinations
- Intra layer feature representation
  ```
  python3 main.py --enable_ilfr True --enable_ssl False --ssl_type None 
  ```

- Semi Supervised learning
  - Unique class case
    ```
    python3 main.py --enable_ilfr True --enable_ssl False --ssl_type unique
    ```
  - Random class case
    ```
    python3 main.py --enable_ilfr True --enable_ssl True --ssl_type None 
    ```

- Intra layer feature representation
  ```
  python3 main.py --enable_ilfr True --enable_ssl True --ssl_type random 
  ```
  
## To cite the paper
  ```bash
  ```
