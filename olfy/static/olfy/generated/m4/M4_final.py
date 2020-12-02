#!/usr/bin/env python
# coding: utf-8

# In[3]:


from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import pandas as pd
import numpy as np
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import math
import random
from captum.attr import IntegratedGradients
import pickle


# In[4]:


def one_hot_smile(smile):
    key="()+–./-0123456789=#@$ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]abcdefghijklmnopqrstuvwxyz^"
    test_list=list(key)
    res = {val : idx  for idx, val in enumerate(test_list)}
    #smile="^"+smile
    smile=smile+("^"*(300-len(smile)))
    array=[[0 for j in range(len(key))] for i in range(300)]
    for i in range(len(smile)):
        array[i][res[smile[i]]]=1
        array=torch.Tensor(array)
    return array


# In[5]:


def one_hot_seq(seq):
    key="ABCDEFGHIJKLMNOPQRSTUVWXYZ^"
    seq=seq.upper()
    test_list=list(key)
    res = {val : idx  for idx, val in enumerate(test_list)}
    #seq="^"+seq+"$"
    seq=seq+("^"*(400-len(seq)))
    array=[[0 for j in range(len(key))] for i in range(400)]
    for i in range(len(seq)):
        array[i][res[seq[i]]]=1
        array=torch.Tensor(array)
    return array


# In[6]:


class BLSTM(nn.Module):
    def __init__(self, input_smile_dim, hidden_smile_dim, layer_smile_dim,input_seq_dim, hidden_seq_dim, layer_seq_dim, output_dim):
        super(BLSTM, self).__init__()
        self.hidden_smile_dim = hidden_smile_dim
        self.layer_smile_dim = layer_smile_dim
        self.hidden_seq_dim = hidden_seq_dim
        self.layer_seq_dim = layer_seq_dim
        self.output_dim = output_dim
        self.smile_len = 300
        self.seq_len = 400
        self.num_smile_dir=2
        self.num_seq_dir=2
        
        self.lstm_smile = nn.LSTM(input_smile_dim, hidden_smile_dim, layer_smile_dim,bidirectional=True)
        self.lstm_seq = nn.LSTM(input_seq_dim, hidden_seq_dim, layer_seq_dim,bidirectional=True)
        self.dropout = nn.Dropout(p=0.1)
        self.fc_smile= nn.Linear(self.smile_len*hidden_smile_dim*self.num_smile_dir,50)
        self.fc_seq= nn.Linear(self.seq_len*hidden_seq_dim*self.num_seq_dir,50)
        
        self.fc_combined = nn.Sequential(nn.Linear(100,10),nn.ReLU(),nn.Linear(10,output_dim))

    def forward(self, x1,x2):
        h0_smile = torch.zeros(self.layer_smile_dim*self.num_smile_dir, x1.size(1), self.hidden_smile_dim).requires_grad_()
        c0_smile = torch.zeros(self.layer_smile_dim*self.num_smile_dir, x1.size(1), self.hidden_smile_dim).requires_grad_()
        h0_seq = torch.zeros(self.layer_seq_dim*self.num_seq_dir, x2.size(1), self.hidden_seq_dim).requires_grad_()
        c0_seq = torch.zeros(self.layer_seq_dim*self.num_seq_dir, x2.size(1), self.hidden_seq_dim).requires_grad_()

        out_smile, (hn_smile, cn_smile) = self.lstm_smile(x1, (h0_smile, c0_smile))
        out_seq, (hn_seq, cn_seq) = self.lstm_seq(x2, (h0_seq, c0_seq))
        out_smile = self.dropout(out_smile)
        out_seq = self.dropout(out_seq)
        out_smile=self.fc_smile(out_smile.view(-1,self.smile_len*self.hidden_smile_dim*self.num_smile_dir))
        out_seq=self.fc_seq(out_seq.view(-1,self.seq_len*self.hidden_seq_dim*self.num_seq_dir))
        out_smile = self.dropout(out_smile)
        out_seq = self.dropout(out_seq)
        #out_combined=torch.cat(out_smile,out_seq)
        out_combined=torch.cat((out_smile,out_seq), 1)
        out_combined=self.fc_combined(out_combined)

        prob=nn.Softmax(dim=1)(out_combined)
        pred=nn.LogSoftmax(dim=1)(out_combined)
        return pred


# In[38]:


def user_predict(model, x_input_smile, x_input_seq,count):
    x_user_smile=one_hot_smile(x_input_smile)
    x_user_smile=list(x_user_smile)
    x_user_smile=torch.stack(x_user_smile)
    x_user_smile=x_user_smile.view(1,300,77)

    x_user_seq=one_hot_seq(x_input_seq)
    x_user_seq=list(x_user_seq)
    x_user_seq=torch.stack(x_user_seq)
    x_user_seq=x_user_seq.view(1,400,27)
    
    scores = model(x_user_smile,x_user_seq)
    _, predictions = scores.max(1)
    pred_ind = predictions.item()
    print("Prediction is ",pred_ind )

    prob=torch.exp(scores)
    prob=prob.tolist()
    print("Probability is",  round(prob[0][predictions.item()],3) )
    z = [predictions.item(),round(prob[0][predictions.item()],3)]

    ig = IntegratedGradients(model)
    x_user_smile.requires_grad_()
    baseline = torch.zeros(1, 300, 77)
    for i in baseline[0]:
        i[-1]=1

    attr,delta= ig.attribute((x_user_smile,x_user_seq), target=1,return_convergence_delta=True)
    attr=attr[0].view(300,77)
    maxattr,_=torch.max(attr,dim=1)
    minattr,_=torch.min(attr,dim=1)
    relevance=maxattr+minattr
    relevance=relevance.detach().numpy()
    data_relevance=pd.DataFrame()
    data_relevance["values"]=relevance

    len_smile=len(x_input_smile)
    
    cropped_smile_relevance=data_relevance.iloc[0:len_smile]
    
    x_smile_labels=pd.Series(list(x_input_smile))
    cropped_smile_relevance['smile_char']=x_smile_labels
    impacts=[]
    
    cropped_smile_relevance['positive']=['']*len_smile
    cropped_smile_relevance['negative']=['']*len_smile
    for row in range(len_smile):
        if (ord(cropped_smile_relevance['smile_char'][row])<65 or ord(cropped_smile_relevance['smile_char'][row])>90):
            cropped_smile_relevance['values'][row]=0
            cropped_smile_relevance['positive'][row]=0
            cropped_smile_relevance['negative'][row]=0
        else:
            if(cropped_smile_relevance['values'][row]>0):
                cropped_smile_relevance['positive'][row]=cropped_smile_relevance['values'][row]
                cropped_smile_relevance['negative'][row]=0
            elif(cropped_smile_relevance['values'][row]<0):
                cropped_smile_relevance['negative'][row]=cropped_smile_relevance['values'][row]
                cropped_smile_relevance['positive'][row]=0
            else:
                cropped_smile_relevance['positive'][row]=0
                cropped_smile_relevance['negative'][row]=0
            impacts.append(cropped_smile_relevance['values'][row])
            
    print(cropped_smile_relevance)
    ax=cropped_smile_relevance.plot( y=["positive", "negative"], color=['green', 'red'], kind="bar", figsize=(25,15))
    ax.legend(['Contribution to Binding', 'Contribution to non binding'])
    ax.set_xticklabels(cropped_smile_relevance['smile_char'],fontsize=15,rotation=0)
    ax.set_xlabel("Smiles",fontsize=15)
    ax.set_ylabel("Relevance",fontsize=15,rotation=0)
    ax.figure.savefig(f'{count}_SmileInterpretability.png')
    impacts=np.array(impacts)
    print(impacts)

#     molecule structure interpretability
    mol=x_input_smile
    m = Chem.MolFromSmiles(mol)
    num_atoms = m.GetNumAtoms()
    labels = [ m.GetAtomWithIdx(i).GetSymbol().upper() for i in range(num_atoms) ]
    colors = {}
    i=0
    k=0
    y_max = np.max(impacts)
    y_min = np.min(impacts)
    dist = y_max - y_min
    while i < len(mol):
        c = mol[i]
        n = ""
        if c.upper() not in "CBONSPFIK":
            print(mol[i], 0.0, "0xFFFFFF")
        else:       
            if i + 1 < len(mol):
                n = mol[i+1]
            sym = c + n    
            sym = sym.strip()
            com = sym.upper()
            if com == "BR" or com == "CL" or com == "NA":
                i = i + 1
            else:
                com = c.upper()
                sym = c
            if com == labels[k]:
                color = "0xBBBBBB"
                triple = [0, 0 ,0]
                if impacts[k] > 0.0:
                    y = int(math.floor(255.0 - 155.0 * impacts[k]  / y_max))
                    color = "0x00" + hex(y)[-2:] + "00"
                    triple[1] = y /255.0
                if impacts[k] < 0.0:
                    y = int(math.floor(255.0 - 155.0 * impacts[k]  / y_min))
                    color = "0x" + hex(y)[-2:] + "0000"
                    triple[0] = y / 255.0
                colors[k]= tuple(triple)
                print(sym, impacts[k], color)
                k = k + 1   
        i = i + 1
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 400)

    drawer.DrawMolecule(m,highlightAtoms = [i for i in range(num_atoms)], highlightBonds=[], highlightAtomColors = colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:','')

    fp = open(f"{count}_mol.svg", "w")
    print(svg, file=fp)
    fp.close()
    
    
#     Sequence Interpretability
    ax=plt.figure()
    baseline = torch.zeros(2, 400, 27)
    ig = IntegratedGradients(model)
    x_user_seq.requires_grad_()
    x_user_smile.requires_grad_()
    attr,delta= ig.attribute((x_user_smile,x_user_seq), target=1,return_convergence_delta=True)
    smile_attr=attr[0].view(300,77)
    seq_attr=attr[1].view(400,27)
    maxattr,_=torch.max(seq_attr,dim=1)
    minattr,_=torch.min(seq_attr,dim=1)
    relevance=maxattr+minattr
    relevance=relevance.detach().numpy()
    data_relevance=pd.DataFrame()
    data_relevance["values"]=relevance

    len_seq=len(x_input_seq)
    cropped_seq_relevance=data_relevance.iloc[0:len_seq]


    x_seq_labels=pd.Series(list(x_input_seq))
    cropped_seq_relevance['seq_char']=x_seq_labels

    cropped_seq_relevance['positive']=['']*len_seq
    cropped_seq_relevance['negative']=['']*len_seq
    
    for row in range(len_seq):
        if (ord(cropped_seq_relevance['seq_char'][row])<65 or ord(cropped_seq_relevance['seq_char'][row])>90):
            cropped_seq_relevance['values'][row]=0
            cropped_seq_relevance['positive'][row]=0
            cropped_seq_relevance['negative'][row]=0
        else:
            if(cropped_seq_relevance['values'][row]>0):
                cropped_seq_relevance['positive'][row]=cropped_seq_relevance['values'][row]
                cropped_seq_relevance['negative'][row]=0
            elif(cropped_seq_relevance['values'][row]<0):
                cropped_seq_relevance['negative'][row]=cropped_seq_relevance['values'][row]
                cropped_seq_relevance['positive'][row]=0
            else:
                cropped_seq_relevance['positive'][row]=0
                cropped_seq_relevance['negative'][row]=0
             
            
     
    print(cropped_seq_relevance)

    ax=cropped_seq_relevance.plot( y=["positive", "negative"], color=['green', 'red'], kind="bar", figsize=(35, 15) )
    ax.legend(['Contribution to Binding', 'Contribution to non binding'])
    ax.set_xticklabels(cropped_seq_relevance['seq_char'],fontsize=15,rotation=0)
    ax.set_xlabel("Receptor Sequence",fontsize=15)
    ax.set_ylabel("Relevance",fontsize=15,rotation=0)
    ax.figure.savefig(f'{count}_SequenceInterpretability.png')
    return z


# In[9]:


filename = 'M4_final.sav'
loaded_model = pickle.load(open(filename, 'rb'))

data = pd.read_csv("input.csv")
output = []
index = data.index
number_of_rows = len(index)
for i in range(number_of_rows):
	temp = []
	user_smile = str(data["smiles"][i])
	user_seq = str(data["seq"][i])
	z = user_predict(loaded_model, user_smile,user_seq,i+1)
	temp.append(user_smile)
	temp.append(user_seq)
	temp.append(z[0])
	temp.append(z[1])
	output.append(temp)

df = pd.DataFrame(output, columns = ['smiles','seq','status','prob'])
df.to_csv("output.csv")


# In[ ]:




