# THGNN-TCP
A novel temporal heterogeneous graph neural networks-based technology convergence prediction framework considering structural and semantic features (THGNN-TCP) for dynamically identifying technology opportunities at multiple granularities.
The framework takes into account the co-occurrence relationship, semantic distance and knowledge flow among technologies, as well as the linkage between science and technology to capture the changes in the internal and external influencing factors of technology. First, the temporal technology heterogeneous networks are constructed using patent and paper as dataset according to time divisions. Then, the MAGNN model was improved to learn the feature representations of the classification number nodes at each time period by introducing a node type-aware strategy and enhancing information communication between attention heads. Subsequently, LSTM is used to identify the latent changes in technology classifications over time. Finally, technology opportunities are predicted through link prediction between the lowest level classification numbers.
## Requirement
* python 3.8.5
* pytorch==1.12.1+cu116
* dgl==1.0.1+cu116
* scipy==1.10.1
* pandas==1.5.3
* networkx==2.5
* numpy==1.20.3
* sklearn


## Usage
Original dataset:
* data_v2.xlsx

Download the original and the dealt dataset from https://drive.google.com/drive/folders/1CkHuzyNrmjgx0JZVm6QvQiwGdEkCzHUK?usp=drive_link

### Training
```python
python train.py
```
### Prediction
```python
python prediction.py
```
