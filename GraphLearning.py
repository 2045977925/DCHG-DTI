import torch
import pandas as pd
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data
from HeteroNetwork import build_heterogeneous_network

class HeteroRGCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_relations):
        super(HeteroRGCN, self).__init__()
        self.conv1 = RGCNConv(num_node_features, hidden_channels, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations=num_relations)
        self.conv3 = RGCNConv(hidden_channels, hidden_channels, num_relations=num_relations)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_type)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_type)
        x = torch.relu(x)
        x = self.dropout(x)

        return x


def get_model(num_node_features, hidden_channels, num_relations):
    # 仅返回R-GCN模型实例
    return HeteroRGCN(num_node_features, hidden_channels, num_relations)


def build_data_from_networkx(G):
    node_to_index = {node: idx for idx, node in enumerate(G.nodes) if 'features' in G.nodes[node]}
    index_to_node = {idx: node for node, idx in node_to_index.items()}

    edge_index = torch.tensor(
        [(node_to_index[u], node_to_index[v]) for u, v in G.edges if u in node_to_index and v in node_to_index],
        dtype=torch.long
    ).t().contiguous()

    edge_type_str = [G[u][v]['type'] for u, v in G.edges if u in node_to_index and v in node_to_index]
    unique_edge_types = {edge_type: idx for idx, edge_type in enumerate(set(edge_type_str))}

    edge_type = torch.tensor([unique_edge_types[et] for et in edge_type_str], dtype=torch.long)

    node_features = torch.tensor(
        [G.nodes[node]['features'] for node in G.nodes if 'features' in G.nodes[node]],
        dtype=torch.float
    )

    return Data(x=node_features, edge_index=edge_index, edge_type=edge_type), unique_edge_types, index_to_node


def separate_features_by_type(output, index_to_node):
    drug_features, protein_features = [], []
    drug_ids, protein_ids = [], []

    for i, output_vector in enumerate(output):
        node_id = index_to_node[i]
        if node_id.startswith('DB'):  # 假设药物ID以'DB'开头
            drug_features.append(output_vector.cpu().numpy())
            drug_ids.append(node_id)
        else:
            protein_features.append(output_vector.cpu().numpy())
            protein_ids.append(node_id)

    print(f"Extracted {len(drug_ids)} drug features and {len(protein_ids)} protein features.")

    return (drug_ids, drug_features), (protein_ids, protein_features)


def save_features_to_csv(ids, features, file_name):
    if features:  # 仅在有特征时保存
        df = pd.DataFrame({'ID': ids, 'Features': list(features)})
        df.to_csv(file_name, index=False)
        print(f"Saved features to {file_name}")
    else:
        print(f"No features to save for {file_name}.")


def main():
    channels = [
        {
            "drug_features_file": "nodefeatures/drug_smiles_features.csv",
            "protein_features_file": "nodefeatures/protein_sequence_features.csv",
            "drug_similarity_file": "data/combined_drug_similarity_matrix.csv",
            "protein_similarity_file": "data/combined_protein_similarity_matrix.csv",
            "interaction_file": "data/drug_protein.csv",
            "drug_id_col": "Drug_ID",
            "protein_id_col": "Protein_ID",
            "features_col": "Features"
        },
        {
            "drug_features_file": "nodefeatures/drug_graph_features.csv",
            "protein_features_file": "nodefeatures/protein_graph_features.csv",
            "drug_similarity_file": "data/combined_drug_similarity_matrix.csv",
            "protein_similarity_file": "data/combined_protein_similarity_matrix.csv",
            "interaction_file": "data/drug_protein.csv",
            "drug_id_col": "ID",
            "protein_id_col": "ID",
            "features_col": "Graph_Features"
        }
    ]

    for channel in channels:
        G = build_heterogeneous_network(
            channel["drug_features_file"],
            channel["protein_features_file"],
            channel["drug_similarity_file"],
            channel["protein_similarity_file"],
            channel["interaction_file"],
            channel["drug_id_col"],
            channel["protein_id_col"],
            channel["features_col"]
        )

        data, unique_edge_types, index_to_node = build_data_from_networkx(G)

        num_node_features = data.x.size(1)
        hidden_channels = 64
        num_relations = len(unique_edge_types)

        model = get_model(num_node_features, hidden_channels, num_relations)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        data = data.to(device)

        model.eval()
        with torch.no_grad():
            # R-GCN需要edge_type
            output = model(data.x, data.edge_index, data.edge_type)

        (drug_ids, drug_features), (protein_ids, protein_features) = separate_features_by_type(output, index_to_node)

        if channel["features_col"] == "Features":
            save_features_to_csv(drug_ids, drug_features, f'Drug_SMILES_Features_RGCN_{channel["features_col"]}.csv')
            save_features_to_csv(protein_ids, protein_features, f'Protein_Sequence_Features_RGCN_{channel["features_col"]}.csv')
        elif channel["features_col"] == "Graph_Features":
            save_features_to_csv(drug_ids, drug_features, f'Drug_Graph_Features_RGCN_{channel["features_col"]}.csv')
            save_features_to_csv(protein_ids, protein_features, f'Protein_Graph_Features_RGCN_{channel["features_col"]}.csv')

        if drug_features:
            print("Drug Features Shape:", len(drug_features), drug_features[0].shape)
        else:
            print("No drug features available.")

        if protein_features:
            print("Protein Features Shape:", len(protein_features), protein_features[0].shape)
        else:
            print("No protein features available.")


if __name__ == '__main__':
    main()
