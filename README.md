# **DCHG-DTI**
 This repository contains the code for paper: Prediction of Drug-Target Interactions Based on Dual-Channel Heterogeneous Graph Neural Networks
![DCHG_Framework](https://github.com/user-attachments/assets/1d3c1a63-3391-4cad-86e7-ce4ca77af628)
 Figure 1. The DCHG-DTI prediction framework is divided into four parts: (A) Different feature extraction modules are used to extract local features of drugs and targets. (B) A heterogeneous graph neural network is constructed based on the local features, and a relation-aware graph convolutional network is employed to extract global features. (C) The MHBA feature fusion module is utilized to fuse the features obtained from (B). (D) The fused feature representations of drugs and proteins are used to predict potential DTIs.
# **Environment Setup**
This project uses Conda to manage the environment to ensure all dependencies are correctly installed. Please follow the steps below to set up the environment:
## *Step 1: Install Anaconda or Miniconda*
If you haven't installed Anaconda or Miniconda yet, please visit the [Anaconda official website](https://www.anaconda.com/products/individual) or [Miniconda official website](https://docs.conda.io/en/latest/miniconda.html) to download and install it.
## *Step 2: Create and Activate Environment*
### Create the environment
conda env create -f environment.yml
### Activate the environment
conda activate DCHGDTI
# **How to run**
## *Step 1: Data Preprocessing*
Run the dataprocess.py script to prepare the one-dimensional embeddings and two-dimensional graph data for drugs and proteins across different channels. This step ensures that your data is properly processed for subsequent local feature extraction.
## *Step 2: Subsequent Local Feature Extraction*
Run the FeatureExtra.py script to extract local features of drugs and proteins in different channels respectively, which will be used for constructing the heterogeneous graph neural network.
## *Step 3: Global Feature Extraction*
Run the GraphLearning.py script to construct the heterogeneous graph neural network and perform global feature extraction of drugs and proteins across different channels.
## *Step 4: Feature Fusion*
Run the FeatureFusion.py script to integrate different features, facilitating subsequent Drug-Target Interaction (DTI) prediction.
## *Step 5: DTI Prediction*
Use the DTIPrediction.py script to apply the fused features for DTI prediction. This step utilizes the features obtained from the previous step to make predictions and generate results. Following these steps in sequence will help ensure successful replication of the results presented in our paper. If you encounter any challenges or require more detailed information during execution, please refer to our code documentation and program instructions for guidance on parameter settings and data preparation.
# Contact
For questions or suggestions, please contact [fcy_0613@163.com].
