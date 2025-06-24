import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import ast
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from dataprocess import prepare_embeddings

# 通道A特征提取模块定义
# 定义 CNN 模型
class Conv1d(nn.Module):
    def __init__(self, input_dim, channel, kernel_size):
        super(Conv1d, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, channel, kernel_size, padding=1)
        self.conv2 = nn.Conv1d(channel, channel, kernel_size, padding=1)
        self.conv3 = nn.Conv1d(channel, channel, kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # 添加 Dropout
        self.globalmaxpool = nn.AdaptiveMaxPool1d(1)
        self.globalavgpool = nn.AdaptiveAvgPool1d(1)  # 添加全局平均池化

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Dropout
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)  # Dropout
        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout(x)  # Dropout

        # 使用全局最大池化和全局平均池化
        max_features = self.globalmaxpool(x)
        avg_features = self.globalavgpool(x)
        x = max_features + avg_features  # 简单相加特征
        x = x.squeeze(-1)
        return x

# 提取特征的函数
def extract_features(embeddings, cnn_model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn_model.to(device)
    embeddings = embeddings.to(device)
    # 确保嵌入是3D张量
    embeddings = embeddings.unsqueeze(1)  # 添加一个额外的维度
    print(f"Embeddings shape before CNN: {embeddings.shape}")
    features = cnn_model(embeddings)
    print(f"Features shape after CNN: {features.shape}")
    return features.cpu().detach()

# 通道B特征提取模块定义
class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

def load_drug_graph_from_csv(csv_file_path):
    df = pd.read_csv(csv_file_path)
    graphs = []

    for index, row in df.iterrows():
        drug_id = row['DRUGBANK_ID']
        node_features = torch.tensor(ast.literal_eval(row['Atom_Features']), dtype=torch.float)
        edge_features = ast.literal_eval(row['Edge_Features'])

        edge_index = torch.tensor([[edge[0], edge[1]] for edge in edge_features], dtype=torch.long).t()

        if edge_index.numel() == 0 or edge_index.size(0) != 2:
            print(f"Skipping Drug {drug_id}: invalid or no edges.")
            continue

        print(f"Drug {drug_id} edge_index shape: {edge_index.shape}")

        graph = Data(x=node_features, edge_index=edge_index)
        graphs.append((drug_id, graph))

    return graphs


def load_protein_graph_from_csv(csv_file_path):
    graphs = []

    for chunk in pd.read_csv(csv_file_path, chunksize=1):  # 每批只处理一个图
        for index, row in chunk.iterrows():
            protein_id = row['Protein_ID']
            adjacency_matrix_str = row['Adjacency_Matrix']

            try:
                # 使用 `ast.literal_eval` 来安全地解析字符串为列表
                adjacency_matrix_list = ast.literal_eval(adjacency_matrix_str)

                # 将解析后的列表转换为NumPy数组
                adjacency_matrix = np.array(adjacency_matrix_list, dtype=float)

                # 检查是否是方阵
                if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
                    print(f"Protein {protein_id} does not form a perfect square matrix, skipping.")
                    continue

                edge_index = np.array(np.where(adjacency_matrix > 0))
                node_count = adjacency_matrix.shape[0]
                node_features = np.eye(node_count, dtype=float)

                x = torch.tensor(node_features, dtype=torch.float)
                edge_index = torch.tensor(edge_index, dtype=torch.long)

                graph = Data(x=x, edge_index=edge_index)
                graphs.append((protein_id, graph))

            except ValueError as e:
                print(f"Error processing protein {protein_id}: {e}")
                continue

    return graphs

def load_protein_graph_from_csv_for_error_processing_protein(csv_file_path, target_protein_ids):
    graphs = []

    for chunk in pd.read_csv(csv_file_path, chunksize=1):
        for index, row in chunk.iterrows():
            protein_id = row['Protein_ID']
            if protein_id not in target_protein_ids:
                continue

            adjacency_matrix_str = row['Adjacency_Matrix']

            try:
                adjacency_matrix_list = ast.literal_eval(adjacency_matrix_str)
                adjacency_matrix = np.array(adjacency_matrix_list, dtype=float)

                if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
                    print(f"Protein {protein_id} does not form a perfect square matrix, skipping.")
                    continue

                # 使用稀疏矩阵表示
                edge_index = np.array(np.where(adjacency_matrix > 0))
                node_count = adjacency_matrix.shape[0]
                node_features = np.eye(node_count, dtype=float)

                x = torch.tensor(node_features, dtype=torch.float)
                edge_index = torch.tensor(edge_index, dtype=torch.long)

                graph = Data(x=x, edge_index=edge_index)
                graphs.append((protein_id, graph))

            except ValueError as e:
                print(f"Error processing protein {protein_id}: {e}")
                continue

    return graphs

def extract_graph_features(graphs, hidden_dim, output_dim):
    results = []

    for entity_id, graph in graphs:
        input_dim = graph.x.size(1)
        model = GraphSAGEModel(input_dim, hidden_dim, output_dim).to('cpu')  # 强制在CPU上运行
        model.eval()
        with torch.no_grad():
            try:
                features = model(graph)
                graph_feature = features.mean(dim=0)
                results.append((entity_id, graph_feature.numpy().tolist()))
                torch.cuda.empty_cache()  # 清理缓存
            except RuntimeError as e:
                print(f"Error processing graph {entity_id}: {e}")
                continue

    return results

def process_graph_features(csv_file, output_file, graph_type, hidden_dim, output_dim):
    if graph_type == 'protein':
        graphs = load_protein_graph_from_csv(csv_file)
    else:
        raise ValueError("Invalid graph_type! Must be 'protein'.")

    features = extract_graph_features(graphs, hidden_dim, output_dim)
    df = pd.DataFrame(features, columns=['ID', 'Graph_Features'])
    df.to_csv(output_file, index=False)
    print(f"{graph_type.capitalize()} features extracted and saved to {output_file}")

# 对error processing protein的特征提取
def extract_graph_features_for_error_processing_protein(graphs, hidden_dim, output_dim):
    results = []

    for entity_id, graph in graphs:
        input_dim = graph.x.size(1)
        model = GraphSAGEModel(input_dim, hidden_dim, output_dim).to('cpu')
        model.eval()

        with torch.no_grad():
            try:
                features = model(graph)
                graph_feature = features.mean(dim=0)
                results.append((entity_id, graph_feature.numpy().tolist()))
            except RuntimeError as e:
                print(f"Error processing graph {entity_id}: {e}")
                continue
            finally:
                del model
                torch.cuda.empty_cache()

    return results

if __name__ == '__main__':
    # 通道A特征提取
    # 文件路径
    protein_file = 'datainput/protein_sequence.csv'
    drug_file = 'datainput/drug_smiles.csv'
    # 读取药物和蛋白质的嵌入
    drugs, proteins, drug_embeddings, protein_embeddings = prepare_embeddings(drug_file, protein_file)
    # 定义CNN模型
    cnn_model = Conv1d(input_dim=128, channel=64, kernel_size=3)
    # 提取药物和蛋白质特征
    drug_features = extract_features(drug_embeddings, cnn_model)
    protein_features = extract_features(protein_embeddings, cnn_model)
    # 保存药物特征
    drug_features_list = drug_features.numpy().tolist()
    drug_data = list(zip(drugs.iloc[:, 0], drug_features_list))  # 使用药物编号
    drug_features_df = pd.DataFrame(drug_data, columns=['Drug_ID', 'Features'])
    drug_features_df.to_csv('nodefeatures/drug_smiles_features.csv', index=False)
    # 保存蛋白质特征
    protein_features_list = protein_features.numpy().tolist()
    protein_data = list(zip(proteins.iloc[:, 0], protein_features_list))  # 使用蛋白质编号
    protein_features_df = pd.DataFrame(protein_data, columns=['Protein_ID', 'Features'])
    protein_features_df.to_csv('nodefeatures/protein_sequence_features.csv', index=False)
    print("Drug features saved to drug_features.csv")
    print("Protein features saved to protein_features.csv")

    # 通道B特征提取
    # 药物特征提取
    process_graph_features(
        'graphdata/drugs/drug_graph.csv',
        'nodefeatures/drug_graph_features.csv',
        'drug',
        hidden_dim=32,
        output_dim=64
    )

    # 蛋白质特征提取
    process_graph_features(
        'graphdata/proteins/protein_graph.csv',
        'nodefeatures/protein_graph_features.csv',
        'protein',
        hidden_dim=32,
        output_dim=64
    )
