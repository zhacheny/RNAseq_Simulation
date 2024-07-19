import os
import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.nn as gnn
import networkx as nx
import numpy as np
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt

# Load synthetic data
def load_synthetic_data(filepath):
    df = pd.read_csv(filepath, index_col=0)
    gene_list = list(df.columns[:-1])
    X = df[gene_list].values
    Y = df['Label'].values
    return X, Y, gene_list

script_directory = os.path.dirname(os.path.abspath(__file__))
data_filepath = os.path.join(script_directory, 'combined_data.csv')

X, Y, gene_list = load_synthetic_data(data_filepath)
print(f"Loaded data with {X.shape[0]} samples and {X.shape[1]} features.")

# Load graph
input_path = "/projects/compsci/karuturi/common-data-dir/ALZ/human/latest_graphml/"
fpath = os.path.join(input_path, "bdFiltered_APP_Metabolism_undirected_filt_node_edge.graphml")
g = nx.read_graphml(fpath)
g = g.to_undirected()

plt.figure()
nx.draw(g)
plt.savefig("g_graphml.png")  
plt.close() 

# Extract gene list and create dictionaries
gene_list_graph = [g.nodes[node]['name'] for node in g.nodes()]
node_id_to_gene_dict = {i: g for i, g in enumerate(gene_list_graph)}
gene_to_node_id_dict = {g: i for i, g in node_id_to_gene_dict.items()}

# Generate edge index
A = nx.to_numpy_array(g)
print(f"Num genes in graph = {len(A)}")
edge_index = np.vstack(list(np.where(A)))
print(f"Edge index shape: {edge_index.shape}")

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)
edge_index = torch.tensor(edge_index, dtype=torch.int64)

# Create DataLoader
dataset = TensorDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class GNN(nn.Module):
    def __init__(self, num_labels=2):
        super(GNN, self).__init__()
        self.conv1 = gnn.GCNConv(1, 50)
        self.conv2 = gnn.GCNConv(50, 50)
        self.clf = nn.Linear(50, num_labels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = gnn.global_mean_pool(x, batch)
        return self.clf(x)

model = GNN()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
num_epochs = 100

for epoch in range(num_epochs):
    epoch_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        x, y = batch
        graphs = [Data(x=x[i].unsqueeze(-1), edge_index=edge_index) for i in range(len(x))]
        # import pdb; pdb.set_trace()
        graph_batch = Batch.from_data_list(graphs)

        optimizer.zero_grad()
        ypred = model(x=graph_batch.x, edge_index=graph_batch.edge_index, batch=graph_batch.batch)
        loss = F.nll_loss(F.log_softmax(ypred, dim=1), y.to(torch.int64))
        loss.backward()
        optimizer.step()

        # print(f'Epoch {epoch}, Loss: {loss.item()}')

        epoch_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(ypred, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    # Calculate average loss and accuracy
    avg_loss = epoch_loss / len(dataloader)
    accuracy = correct / total

    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')


