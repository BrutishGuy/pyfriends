# pyfriends
Python implementation of the paper "PyFriends: The First Fully Generalized Friends-of-Friends Extragalactic Galaxy Group Finder", using a Friends-of-Friends (FoF) algorithm for galaxy group detection, augmented by graph theory approaches. 

In this repository we do the graph theoretic things, such that the galactic friendships can be made to be done. 

A detailed description of the algorithm can be found in the paper above linked on ArXiv.org.

## Installation

Download the repository through Git (For Windows, you can download Git Bash For Windows here).

`git clone https://github.com/BrutishGuy/pyfriends.git`

## Data

Example data has been included in the ./data/ folder of this repository. It follows from [Macri et al.](https://ui.adsabs.harvard.edu/abs/2012yCat..21990026H/abstract) 

## Execution

To execute the code, one must modify the config.text file to set necessary parameters for the run. These are already set to reasonable parameters.

Detailed explanation on these parameters will follow.

To run the code, simply execute the file Py2Friends.py through the command line or your favourite editor, ensuring your working directory is set to the repository directory, such that config.txt is in your working directory.
Then, simply run 
```python
python ./code/Py2Friends.py
```

For any issues or feature requests, please log an issue on this Github repository.
