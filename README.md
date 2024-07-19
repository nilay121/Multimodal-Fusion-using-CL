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
  
## To cite the paper
  ```bash
  ```
