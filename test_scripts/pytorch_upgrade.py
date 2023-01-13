# TODO:
# import torch
# tensor = torch.ones(4, 4)
# print(f"Device tensor is stored on: {tensor.device}")

# # We move our tensor to the GPU if available
# if torch.cuda.is_available():
#   tensor = tensor.to('cuda')
#   print(f"Device tensor is stored on: {tensor.device}")

import time
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import load_iris
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torchmetrics import R2Score, ExplainedVariance, MeanAbsolutePercentageError

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(92, 1024)
        self.do1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 1024)
        self.do2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.do1(x)
        x = F.relu(self.fc2(x))
        x = self.do2(x)
        x = F.leaky_relu(self.fc3(x))
        return x

class sk8r_lite():

    def __init__(self, filename):
        f = open(filename)
        loaded = json.load(f)
        self.start = 10
        self.response = "goals"
        self.id = loaded["id"]
        self.name = loaded["name"]
        self.data = pd.DataFrame.from_dict(json.loads(loaded["data"]))
        self.model = None
        self.set_columns()
        self.generate_data()
        # print(self.X.shape)
        self.train_model()

    def set_columns(self):
        self.numX = [
            "g_toi", "a_toi", "s_toi", "h_toi", "ga_toi", "ta_toi",
            "gt3", "at3", "st3", "ht3", "gat3", "tat3",
            "gt5", "at5", "st5", "ht5", "gat5", "tat5",
            "gt10", "at10", "st10", "ht10", "gat10", "tat10",
            "tsgt1", "tsat1", "tsst1", "tstat1", "tsgat1",
            "osgat1", "ostat1", "osbt1", "osht1", "odbt1", "odht1",
            "oggat1", "ogsat1", "ogsvp1",
            "tsgt3", "tsat3", "tsst3", "tstat3", "tsgat3",
            "osgat3", "ostat3", "osbt3", "osht3", "odbt3", "odht3",
            "oggat3", "ogsat3", "ogsvp3",
            "tsgt5", "tsat5", "tsst5", "tstat5", "tsgat5",
            "osgat5", "ostat5", "osbt5", "osht5", "odbt5", "odht5",
            "oggat5", "ogsat5", "ogsvp5",
            "tsgt10", "tsat10", "tsst10", "tstat10", "tsgat10",
            "osgat10", "ostat10", "osbt10", "osht10", "odbt10", "odht10",
            "oggat10", "ogsat10", "ogsvp10"
        ]
        self.catX = [["is_home", 2], ["tdr", 5], ["odr", 5]]

    def generate_data(self):
        
        # Set Responses
        idx = self.data.index
        self.Y = self.data.loc[idx[self.start+1]:, self.response].to_numpy()

        # Set-up Data
        self.X = self.data.loc[idx[self.start]:idx[len(idx)-2], self.numX].to_numpy()
        for factor in self.catX:
            catcol = self.data.loc[idx[self.start+1]:, factor[0]].to_numpy().astype(int)
            to_onehot = np.zeros((catcol.size, factor[1]))
            to_onehot[np.arange(catcol.size), catcol] = 1
            self.X = np.concatenate((self.X, to_onehot), axis=1)

    def train_model(self):

        # Setup Parameters
        epochs = 3000
        self.model = Net()
        X, Y = torch.Tensor(self.X), torch.Tensor(self.Y)
        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        ll = nn.HuberLoss(delta=1.0)
        oo = optim.RMSprop(self.model.parameters(), lr=1e-4)

        # Train the Neural Network
        for epoch in range(epochs):

            running_loss = 0.0
            for i, data in enumerate(loader, 0):
                
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                oo.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = ll(outputs, labels)
                loss.backward()
                oo.step()

                # print statistics
                running_loss += loss.item()

            if epoch % 100 == 999:    # print every 100 epochs
                print(f'[{epoch + 1}] loss: {running_loss / i}')
                running_loss = 0.0

        print('Finished Training')
        Y = torch.reshape(Y, (Y.shape[0],1))
        preds = self.model(X)
        r2score = R2Score()
        mean_abs_percentage_error = MeanAbsolutePercentageError() 
        explained_variance = ExplainedVariance()
        print(f"R2 Score: {r2score(preds, Y)} | Mean Abs % Error: {mean_abs_percentage_error(preds, Y)} | Explained Var: {explained_variance(preds, Y)}")



# Test on guentzel
start = time.time()
sk8r = sk8r_lite('data/Jake Guentzel.json')
end = time.time()
print(f"Time: {end - start}")