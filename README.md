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

## Ros dependencies


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
  
## To replicate the application on a gripper
  - Make sure the Arduino board of the control box and the sensors are connected to the proper ports
  - Train the SynapNet algorithm on the offline data incrementally
    ```bash
    python3 main.py --Uk_classExpPhase False --pseudo_exp False
    ```
  - To perform real-time dynamic training on new unseen objects
    ```bash
    python main.py --Uk_classExpPhase True --pseudo_exp False
    ```
  - To perform a pseudo-real-time experiment
    ```bash
    python main.py --Uk_classExpPhase True --pseudo_exp True
    ```
  
## To cite the paper
  ```bash
  ```
