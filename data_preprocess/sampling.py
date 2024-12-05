import json
import torch
from torch_geometric.data import Data
import torch_geometric.utils
import networkx as nx
from tqdm import tqdm
from torch_geometric.utils import to_networkx
import random

def load_data(jsonl_file):
    """
    Load data from a JSONL file and return a list of dictionaries.
    """
    with open(jsonl_file, 'r') as file:
        data = [json.loads(line) for line in file]
    return data


def build_graph(data):
    """
    Build a NetworkX graph from the data, only including edges where both nodes are center papers.
    """
    G = nx.DiGraph()
    # First, determine all valid papers (that can be center nodes)
    papers = set(entry['paper'] for entry in data)
    # print(len(papers))

    # Only add nodes and edges for papers and their references if the references are also center papers
    for entry in tqdm(data):
        paper = entry['paper']
        if paper in papers:
            G.add_node(paper)
            for ref in entry.get('reference', []):
                # print(ref)
                if ref in papers:  # Only add the reference if it is a center paper
                    G.add_edge(paper, ref)
    print(G.number_of_nodes())
    return G

def convert_to_pyg(G, node_to_idx):
    """
    Convert a NetworkX graph to a PyTorch Geometric graph using a node index mapping.
    """
    # Prepare edge index list by mapping node names to indices
    edge_list = []
    for source, target in G.edges():
        if source in node_to_idx and target in node_to_idx:
            edge_list.append([node_to_idx[source], node_to_idx[target]])    
    
    if not edge_list:
        print("No valid edges to convert to PyG.")
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    num_nodes = len(node_to_idx)  # Explicitly count all nodes, ensuring isolated ones are included
    # print("Node indices:", list(node_to_idx.values()))
    
    return Data(edge_index=edge_index, num_nodes=num_nodes)


def record_neighbors(G, node_to_idx):

    output = {}

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # graph = graph.to(device)
    neighbor_samples = {}
    # G = to_networkx(graph, to_undirected=True)
    pagerank_scores = nx.pagerank(G)

    i = 0
    c = 2

    # Record 1-hop and 2-hop neighbors
    for node in tqdm(G.nodes()):
        one_hop_neighbors = list(nx.neighbors(G, node))
        two_hop_neighbors = list(set(
            neighbor
            for one_hop in one_hop_neighbors
            for neighbor in nx.neighbors(G, one_hop)
        ) - {node} - set(one_hop_neighbors))


        output[node] = {
            'paper_index': node,
            'pr': pagerank_scores[node],
            '1-hop': one_hop_neighbors,
            '2-hop': two_hop_neighbors
        }

        

    return output


def sample_neighbors(data, G, graph, node_to_idx, idx_to_node, num_1hop=5, num_2hop=5):
    """
    Sample neighbors using PyTorch Geometric's efficient utilities, utilizing direct node identifiers.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)
    neighbor_samples = {}
    # G = to_networkx(graph, to_undirected=True)
    pagerank_scores = nx.pagerank(G)  # Calculate PageRank in NetworkX
    # print(pagerank_scores)

    # for entry in tqdm(data):
    for entry in tqdm(data):
        paper = entry['paper']
        node_idx = torch.tensor([node_to_idx[paper]], device=device)

        # 1-hop neighbors
        one_hop_nodes, _, _, _ = torch_geometric.utils.k_hop_subgraph(node_idx, 1, graph.edge_index, num_nodes=graph.num_nodes)
        one_hop_neighbors = [idx_to_node[n.item()] for n in one_hop_nodes if idx_to_node[n.item()] != paper]
        random.shuffle(one_hop_neighbors)  # Shuffle before sampling
        one_hop_neighbors = one_hop_neighbors[:num_1hop]
        print(one_hop_neighbors)
        # one_hop_neighbors = one_hop_neighbors[:num_1hop]  # Limit to num_1hop

        # 2-hop neighbors
        all_two_hop_neighbors = []
        sampled_one_hop = one_hop_neighbors[:min(len(one_hop_neighbors), num_1hop)]
        for neighbor_id in sampled_one_hop:
            neighbor_idx = torch.tensor([node_to_idx[neighbor_id]], device=device)
            two_hop_nodes, _, _, _ = torch_geometric.utils.k_hop_subgraph(neighbor_idx, 1, graph.edge_index, num_nodes=graph.num_nodes)
            two_hop_neighbors = [idx_to_node[n.item()] for n in two_hop_nodes if idx_to_node[n.item()] != neighbor_id and idx_to_node[n.item()] != paper]
            random.shuffle(two_hop_neighbors)  # Shuffle before sampling
            all_two_hop_neighbors.extend(two_hop_neighbors)
        
        all_two_hop_neighbors = list(set(all_two_hop_neighbors))[:num_2hop]  # Limit to num_2hop
        print(all_two_hop_neighbors)

        # Collect all neighbors
        sampled_neighbors = one_hop_neighbors + all_two_hop_neighbors


        num_required = 30
        if len(sampled_neighbors) < num_required:
            sorted_nodes = sorted(pagerank_scores, key=lambda x: pagerank_scores[x], reverse=True)
            additional_neighbors = [n for n in sorted_nodes if n not in sampled_neighbors and n != paper][:num_required - len(sampled_neighbors)]
            sampled_neighbors.extend(additional_neighbors)

        neighbor_samples[paper] = sampled_neighbors
        # print

    return neighbor_samples

def print_pyg_data(graph):
    """
    Print the contents of a PyTorch Geometric Data object.
    """
    print("PyTorch Geometric Graph Data:")
    print("----------------------------")
    # Print edge index
    if graph.edge_index.size(1) > 0:
        print("Edge Index:")
        for i in range(graph.edge_index.size(1)):
            print(f"({graph.edge_index[0, i].item()}, {graph.edge_index[1, i].item()})")
    else:
        print("No edges in the graph.")

    # Print the number of nodes
    print(f"Total number of nodes: {graph.num_nodes}")



data_dir = '/gpfsnyu/scratch/qz2086/AdaGLM/data'
sub_domain = 'Mathematics'

jsonl_file = f'{data_dir}/{sub_domain}/papers.json'
data = load_data(jsonl_file)
G = build_graph(data)
idx_to_node = {idx: node for idx, node in enumerate(G.nodes())}
node_to_idx = {node: idx for idx, node in idx_to_node.items()}
# print(node_to_idx)
graph_pyg = convert_to_pyg(G, node_to_idx)


rec = record_neighbors(G, node_to_idx)
# print(rec)
output_file = f'{data_dir}/{sub_domain}/pr_neighbors.json'
# output_file = '/scratch/qz2086/AdaGLM/data/Mathematics/neighbor_samples.json'
with open(output_file, 'w') as file:
    json.dump(rec, file)