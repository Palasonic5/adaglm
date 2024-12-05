import json
import numpy as np
import networkx as nx
from scipy import sparse as sp
from scipy.sparse.linalg import eigs
import torch

print('calculating lap encoding for math', flush = True)
print('Calculating Laplacian encoding for the domain', flush=True)

def laplacian_positional_encoding(G, pos_enc_dim):
    """
    Graph positional encoding via Laplacian eigenvectors.
    """
    L = nx.normalized_laplacian_matrix(G)  # Calculate the normalized Laplacian matrix
    eigenvalues, eigenvectors = eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2)  # Compute eigenvalues and eigenvectors
    eigenvectors = eigenvectors[:, eigenvalues.argsort()]  # Sort by increasing eigenvalue
    return eigenvectors[:, 1:pos_enc_dim+1].real  # Exclude the first eigenvector

def build_graph_from_json(jsonl_file):
    """
    Build a graph from a JSONL file using NetworkX and track paper IDs.
    """
    G = nx.Graph()
    paper_to_node = {}
    node_to_paper = {}  # To keep track of paper IDs for nodes

    with open(jsonl_file, 'r') as file:
        data = [json.loads(line) for line in file]

    for idx, entry in enumerate(data):
        paper_id = entry['paper']
        paper_to_node[paper_id] = idx
        node_to_paper[idx] = paper_id  # Reverse map to save paper IDs later
        G.add_node(idx, paper_index=paper_id)

    for entry in data:
        src_node = paper_to_node[entry['paper']]
        references = entry.get('reference', [])
        for ref in references:
            if ref in paper_to_node:
                dst_node = paper_to_node[ref]
                G.add_edge(src_node, dst_node)

    return G, node_to_paper

# File paths and domain specifics
jsonl_file = '/gpfsnyu/scratch/qz2086/AdaGLM/data/Mathematics/papers.json'
graph, node_to_paper = build_graph_from_json(jsonl_file)
print(f"Number of nodes in the graph: {graph.number_of_nodes()}")

# Calculate and save embeddings
pos_enc_dim = 768
pos_encodings = laplacian_positional_encoding(graph, pos_enc_dim)
# print(pos_encodings.tensor)
pos_enc_tensor = torch.tensor(pos_encodings, dtype=torch.float32)
print(pos_enc_tensor.shape)
num_nodes = graph.number_of_nodes()
paper_ids_tensor = torch.tensor([int(node_to_paper[i]) for i in range(num_nodes)], dtype=torch.long)
print("First three paper IDs:")
for i in range(3):
    print(paper_ids_tensor[i].item())

# print(paper_ids_tensor)
# Save the embeddings and paper IDs
torch.save({
    "paper_ids": paper_ids_tensor,
    "positional_encodings": pos_enc_tensor
}, 'laplacian_encodings_math.pt')

print("Saved positional encodings and paper IDs ")

