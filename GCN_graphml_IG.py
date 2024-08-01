import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as gnn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from captum.attr import IntegratedGradients

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

# Shuffle the data X, label Y
indices = np.random.permutation(len(X))
X = X[indices]
Y = Y[indices]

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
assert set(gene_list) == set(gene_list_graph), "Gene list from data and graph do not match!"

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
edge_index = torch.tensor(edge_index, dtype=torch.int64)  # Ensure edge_index is int64

# Split the data into the train / valid sets (7:3)
dataset = TensorDataset(X_tensor, Y_tensor)
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(1, 50)
        self.conv2 = GCNConv(50, 50)
        self.clf = nn.Linear(50, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = gnn.global_mean_pool(x, batch)
        x = self.clf(x)
        return torch.sigmoid(x)  # Use sigmoid function, output shape is (num_samples, 1)

loss_func = nn.BCELoss()

model = GNN()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
num_epochs = 1

best_val_auc = 0
best_model_path = 'best_model.pth'
best_val_loss = float('inf')
patience = 5
trigger_times = 0

# Prepare to log the training process
csv_file = open('training_log.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Epoch', 'Train Loss', 'Train Accu', 'Train AUCROC', 'Val Loss', 'Val Accu', 'Val AUCROC'])

# Start training process
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_scores = []

    for batch in train_loader:
        x, y = batch
        graphs = [Data(x=x[i].unsqueeze(1), edge_index=edge_index) for i in range(len(x))]
        graph_batch = Batch.from_data_list(graphs)

        optimizer.zero_grad()
        ypred = model(x=graph_batch.x, edge_index=graph_batch.edge_index, batch=graph_batch.batch)
        loss = loss_func(ypred, y.float().unsqueeze(1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        y_true.extend(y.cpu().numpy())
        y_scores.extend(ypred.detach().cpu().numpy())

        # Calculate accuracy
        predicted = (ypred.detach().cpu().numpy() > 0.5).astype(int)
        total += y.size(0)
        correct += (predicted == y.cpu().numpy()).sum()

    train_loss = epoch_loss / len(train_loader)
    train_accuracy = correct / total
    train_auc = roc_auc_score(y_true, y_scores)

    # Validation process
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    y_true_val = []
    y_scores_val = []
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            graphs = [Data(x=x[i].unsqueeze(1), edge_index=edge_index) for i in range(len(x))]
            graph_batch = Batch.from_data_list(graphs)

            ypred = model(x=graph_batch.x, edge_index=graph_batch.edge_index, batch=graph_batch.batch)
            loss = loss_func(ypred, y.float().unsqueeze(1))

            val_loss += loss.item()

            # Collect true labels and scores for AUC calculation
            y_true_val.extend(y.cpu().numpy())
            y_scores_val.extend(ypred.detach().cpu().numpy())

            # Calculate accuracy
            predicted = (ypred.detach().cpu().numpy() > 0.5).astype(int)
            val_total += y.size(0)
            val_correct += (predicted == y.cpu().numpy()).sum()

    val_loss_avg = val_loss / len(val_loader)
    val_accuracy = val_correct / val_total
    val_auc = roc_auc_score(y_true_val, y_scores_val)

    # Log the metrics for each epoch to CSV
    csv_writer.writerow([epoch + 1, train_loss, train_accuracy, train_auc, val_loss_avg, val_accuracy, val_auc])

    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train AUC: {train_auc:.4f}, Val Loss: {val_loss_avg:.4f}, Val Accuracy: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}')

    # Check for early stopping
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break;

    # Save the model if it has the best AUC-ROC score so far
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), best_model_path)
        print(f'Saved best model at epoch {epoch+1} with Val AUC: {val_auc:.4f}')

# Load the best model
model.load_state_dict(torch.load(best_model_path))
print(f'Loaded best model with Val AUC: {best_val_auc:.4f}')
print(f'trigger_times:{trigger_times}')

# Close the CSV file
csv_file.close()

# Integrated Gradients calculation

# Define a forward function for IG
def forward_fn(mask, x, edge_index, batch):
    # print(mask.shape, x.shape)
    x_masked = torch.einsum("ij, j -> ij", x, mask[0])  # Correcting einsum usage
    graphs = [Data(x=x_masked[i].unsqueeze(1), edge_index=edge_index) for i in range(len(x))]
    graph_batch = Batch.from_data_list(graphs)
    return model(x=graph_batch.x, edge_index=graph_batch.edge_index, batch=graph_batch.batch)

ig = IntegratedGradients(forward_func=forward_fn)

# Prepare the inputs for IG
num_samples = X_tensor.shape[0]
num_features = X_tensor.shape[1]
mask = torch.ones(1, num_features)  # Shape: (1, num_features)
baselines = torch.zeros(1, num_features)  # Shape: (1, num_features)
batch = torch.zeros(num_samples, dtype=torch.long)  # Shape: (num_samples,)
additional_forward_args= (X_tensor, edge_index, batch)
# Calculate IG
attr = ig.attribute(
    inputs= mask, 
    baselines= baselines,
    additional_forward_args=additional_forward_args,
    target=0,
    internal_batch_size=1,
    n_steps=50
)[0]

# Print the attribution results
print(attr)
