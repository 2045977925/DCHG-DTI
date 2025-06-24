import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义多头双线性注意力机制
class MultiHeadBilinearAttention(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(MultiHeadBilinearAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert input_dim % num_heads == 0, "输入维度必须能够被注意力头的数量整除"

        self.bilinear_heads = nn.ModuleList([
            nn.Bilinear(self.head_dim, self.head_dim, 1) for _ in range(num_heads)
        ])

        self.drug_transform = nn.Linear(input_dim, input_dim)
        self.protein_transform = nn.Linear(input_dim, input_dim)

        self.fusion = nn.Linear(input_dim, input_dim)

    def forward(self, drug_features, protein_features):
        drug_features = self.drug_transform(drug_features)
        protein_features = self.protein_transform(protein_features)

        drug_heads = torch.chunk(drug_features, self.num_heads, dim=-1)
        protein_heads = torch.chunk(protein_features, self.num_heads, dim=-1)

        fused_heads = []
        for i, bilinear in enumerate(self.bilinear_heads):
            attention_weights = bilinear(drug_heads[i], protein_heads[i])
            attention_weights = F.softmax(attention_weights, dim=0)
            fused_head = attention_weights * drug_heads[i] * protein_heads[i]
            fused_heads.append(fused_head)

        fused_feature = torch.cat(fused_heads, dim=-1)
        fused_feature = self.fusion(fused_feature)
        return fused_feature


def parse_features(feature_string):
    # 将字符串转换为合法的列表格式
    feature_string = feature_string.replace('[', '').replace(']', '')  # 去掉方括号
    feature_list = [float(x.strip().rstrip(',')) for x in feature_string.split() if x.strip()]  # 去掉逗号并转换为浮点数
    return np.array(feature_list)



def fuse_features(input_file1, input_file2, output_file, input_dim, num_heads=4):
    df1 = pd.read_csv(input_file1)
    df2 = pd.read_csv(input_file2)

    df1['Features'] = df1['Features'].apply(parse_features)
    df2['Features'] = df2['Features'].apply(parse_features)

    common_ids = set(df1['ID']).intersection(set(df2['ID']))
    attention_layer = MultiHeadBilinearAttention(input_dim=input_dim, num_heads=num_heads)

    fused_features = []

    for common_id in common_ids:
        feature1 = df1.loc[df1['ID'] == common_id, 'Features'].values[0]
        feature2 = df2.loc[df2['ID'] == common_id, 'Features'].values[0]

        # 确保特征是二维的
        feature1_tensor = torch.tensor(feature1, dtype=torch.float32).unsqueeze(0)
        feature2_tensor = torch.tensor(feature2, dtype=torch.float32).unsqueeze(0)

        # 使用注意力层进行融合
        fused_feature = attention_layer(feature1_tensor, feature2_tensor).detach().numpy().squeeze()

        # 确保fused_feature是一维的，与common_id连接
        entry = np.concatenate([[common_id], fused_feature])
        fused_features.append(entry)

    # 定义列名
    columns = ['ID'] + [str(i) for i in range(fused_feature.size)]
    fused_features_df = pd.DataFrame(fused_features, columns=columns)
    fused_features_df.to_csv(output_file, index=False)

def final_feature_fusion(drug_fusion_file, protein_fusion_file, interaction_file, output_file, input_dim, num_heads=4):
    drug_features_df = pd.read_csv(drug_fusion_file)
    protein_features_df = pd.read_csv(protein_fusion_file)

    # 将每行的特征重新组合为一个列表
    drug_features_df['Features'] = drug_features_df[[str(i) for i in range(input_dim)]].values.tolist()
    protein_features_df['Features'] = protein_features_df[[str(i) for i in range(input_dim)]].values.tolist()

    interaction_matrix = pd.read_csv(interaction_file, index_col=0)

    common_drugs = set(drug_features_df['ID']).intersection(set(interaction_matrix.index))
    common_proteins = set(protein_features_df['ID']).intersection(set(interaction_matrix.columns))

    attention_layer = MultiHeadBilinearAttention(input_dim=input_dim, num_heads=num_heads)
    features_labels = []

    for drug_id in common_drugs:
        for protein_id in common_proteins:
            interaction = interaction_matrix.loc[drug_id, protein_id]

            drug_feature = drug_features_df.loc[drug_features_df['ID'] == drug_id, 'Features'].values[0]
            protein_feature = protein_features_df.loc[protein_features_df['ID'] == protein_id, 'Features'].values[0]

            # 确保特征是二维的
            drug_feature_tensor = torch.tensor(drug_feature, dtype=torch.float32).unsqueeze(0)
            protein_feature_tensor = torch.tensor(protein_feature, dtype=torch.float32).unsqueeze(0)

            fused_feature = attention_layer(drug_feature_tensor, protein_feature_tensor).detach().numpy().squeeze()

            # 确保fused_feature是一维的，与药物-蛋白质对标识符连接
            entry = np.concatenate([[f'{drug_id}-{protein_id}'], fused_feature, [interaction]])
            features_labels.append(entry)

    columns = ['ID'] + [str(i) for i in range(input_dim)] + ['Interaction']
    features_labels_df = pd.DataFrame(features_labels, columns=columns)

    features_labels_df.to_csv(output_file, index=False)

def fuse_channel_one_features(drug_features_file, protein_features_file, interaction_file, output_file, input_dim, num_heads=4):
    # 读取药物和蛋白质特征
    drug_features_df = pd.read_csv(drug_features_file)
    protein_features_df = pd.read_csv(protein_features_file)

    # 解析特征
    drug_features_df['Features'] = drug_features_df['Features'].apply(parse_features)
    protein_features_df['Features'] = protein_features_df['Features'].apply(parse_features)

    # 读取相互作用矩阵
    interaction_matrix = pd.read_csv(interaction_file, index_col=0)

    # 找到共同的药物和蛋白质
    common_drugs = set(drug_features_df['ID']).intersection(set(interaction_matrix.index))
    common_proteins = set(protein_features_df['ID']).intersection(set(interaction_matrix.columns))

    attention_layer = MultiHeadBilinearAttention(input_dim=input_dim, num_heads=num_heads)
    features_labels = []

    for drug_id in common_drugs:
        for protein_id in common_proteins:
            interaction = interaction_matrix.loc[drug_id, protein_id]

            drug_feature = drug_features_df.loc[drug_features_df['ID'] == drug_id, 'Features'].values[0]
            protein_feature = protein_features_df.loc[protein_features_df['ID'] == protein_id, 'Features'].values[0]

            # 确保特征是二维的
            drug_feature_tensor = torch.tensor(drug_feature, dtype=torch.float32).unsqueeze(0)
            protein_feature_tensor = torch.tensor(protein_feature, dtype=torch.float32).unsqueeze(0)

            fused_feature = attention_layer(drug_feature_tensor, protein_feature_tensor).detach().numpy().squeeze()

            # 确保fused_feature是一维的，与药物-蛋白质对标识符连接
            entry = np.concatenate([[f'{drug_id}-{protein_id}'], fused_feature, [interaction]])
            features_labels.append(entry)

    columns = ['ID'] + [str(i) for i in range(input_dim)] + ['Interaction']
    features_labels_df = pd.DataFrame(features_labels, columns=columns)

    features_labels_df.to_csv(output_file, index=False)


def dif_feature_fusion_one_channel(drug_fusion_file, protein_fusion_file, interaction_file,
                       output_file, input_dim, num_heads=4):
    # 读取药物特征文件
    drug_features_df = pd.read_csv(drug_fusion_file)

    # 解析特征字符串
    drug_features_df['Features'] = drug_features_df['Features'].apply(parse_features)

    # 读取蛋白质特征文件
    protein_features_df = pd.read_csv(protein_fusion_file)

    # 解析蛋白质特征字符串
    protein_features_df['Features'] = protein_features_df['Features'].apply(parse_features)

    # 读取交互矩阵
    interaction_matrix = pd.read_csv(interaction_file, index_col=0)

    # 找到共同的药物和蛋白质
    common_drugs = set(drug_features_df['ID']).intersection(set(interaction_matrix.index))
    common_proteins = set(protein_features_df['ID']).intersection(set(interaction_matrix.columns))

    attention_layer = MultiHeadBilinearAttention(input_dim=input_dim, num_heads=num_heads)
    features_labels = []

    # 遍历共同的药物和蛋白质
    for drug_id in common_drugs:
        for protein_id in common_proteins:
            interaction = interaction_matrix.loc[drug_id, protein_id]

            # 获取药物和蛋白质特征
            drug_feature = drug_features_df.loc[drug_features_df['ID'] == drug_id, 'Features'].values[0]
            protein_feature = protein_features_df.loc[protein_features_df['ID'] == protein_id, 'Features'].values[0]

            # 确保特征是二维的
            drug_feature_tensor = torch.tensor(drug_feature, dtype=torch.float32).unsqueeze(0)  # 增加batch维度
            protein_feature_tensor = torch.tensor(protein_feature, dtype=torch.float32).unsqueeze(0)

            # 使用注意力层进行融合
            fused_feature = attention_layer(drug_feature_tensor, protein_feature_tensor).detach().numpy().squeeze()

            # 确保fused_feature是一维的，与药物-蛋白质对标识符连接
            entry = np.concatenate([[f'{drug_id}-{protein_id}'], fused_feature, [interaction]])
            features_labels.append(entry)

    # 创建新的DataFrame
    columns = ['ID'] + [str(i) for i in range(fused_feature.size)] + ['Interaction']
    features_labels_df = pd.DataFrame(features_labels, columns=columns)

    # 保存结果到文件
    features_labels_df.to_csv(output_file, index=False)
    print(f"Fused features saved to '{output_file}'.")


def dif_feature_fusion(drug_fusion_file, protein_fusion_file, interaction_file,
                   output_file, input_dim, num_heads=4):
    # 读取药物特征文件，假设格式为：ID, Features
    drug_features_df = pd.read_csv(drug_fusion_file)

    # 解析药物特征字符串
    drug_features_df['Features'] = drug_features_df['Features'].apply(parse_features)

    # 读取蛋白质特征文件，假设格式为：ID, 0, 1, ..., 63
    protein_features_df = pd.read_csv(protein_fusion_file)

    # 提取特征列 (假设特征列为第2到第65列)
    protein_features_df['Features'] = protein_features_df.iloc[:, 1:65].values.tolist()

    # 读取交互矩阵
    interaction_matrix = pd.read_csv(interaction_file, index_col=0)

    # 找到共同的药物和蛋白质
    common_drugs = set(drug_features_df['ID']).intersection(set(interaction_matrix.index))
    common_proteins = set(protein_features_df['ID']).intersection(set(interaction_matrix.columns))

    attention_layer = MultiHeadBilinearAttention(input_dim=input_dim, num_heads=num_heads)
    features_labels = []

    # 遍历共同的药物和蛋白质
    for drug_id in common_drugs:
        for protein_id in common_proteins:
            interaction = interaction_matrix.loc[drug_id, protein_id]

            # 获取药物和蛋白质特征
            drug_feature = drug_features_df.loc[drug_features_df['ID'] == drug_id, 'Features'].values[0]
            protein_feature = protein_features_df.loc[protein_features_df['ID'] == protein_id, 'Features'].values[0]

            # 确保特征是二维的
            drug_feature_tensor = torch.tensor(drug_feature, dtype=torch.float32).unsqueeze(0)  # 增加batch维度
            protein_feature_tensor = torch.tensor(protein_feature, dtype=torch.float32).unsqueeze(0)

            # 使用注意力层进行融合
            fused_feature = attention_layer(drug_feature_tensor, protein_feature_tensor).detach().numpy().squeeze()

            # 确保fused_feature是一维的，与药物-蛋白质对标识符连接
            entry = np.concatenate([[f'{drug_id}-{protein_id}'], fused_feature, [interaction]])
            features_labels.append(entry)

    # 创建新的DataFrame
    columns = ['ID'] + [str(i) for i in range(fused_feature.size)] + ['Interaction']
    features_labels_df = pd.DataFrame(features_labels, columns=columns)

    # 保存结果到文件
    features_labels_df.to_csv(output_file, index=False)
    print(f"Fused features saved to '{output_file}'.")

def fuse_channels_AB_features(channelA_file, channelB_file, output_file, input_dim, num_heads=4):
    """
    利用多头双线性注意力机制，将通道A和通道B的融合特征进行进一步融合。

    参数:
        channelA_file: str，通道A融合特征文件路径，格式为(ID, 特征..., Interaction)
        channelB_file: str，通道B融合特征文件路径，格式同上
        output_file: str，融合后结果保存路径
        input_dim: int，单通道特征维度（注意是单通道的）
        num_heads: int，多头数量
    """
    import pandas as pd
    import torch
    import numpy as np

    # 读取两个通道的融合特征文件
    df_A = pd.read_csv(channelA_file)
    df_B = pd.read_csv(channelB_file)

    # 确保ID列是字符串格式，方便匹配
    df_A['ID'] = df_A['ID'].astype(str)
    df_B['ID'] = df_B['ID'].astype(str)

    # 取两个文件中共同的ID（药物-蛋白质对）
    common_ids = set(df_A['ID']).intersection(set(df_B['ID']))

    # 创建多头双线性注意力层，输入维度是单通道的特征维度
    attention_layer = MultiHeadBilinearAttention(input_dim=input_dim, num_heads=num_heads)

    fused_features = []

    for id_ in common_ids:
        # 取通道A特征向量（除ID和Interaction列）
        feat_A = df_A.loc[df_A['ID'] == id_, df_A.columns.difference(['ID', 'Interaction'])].values[0].astype(np.float32)
        # 取通道B特征向量
        feat_B = df_B.loc[df_B['ID'] == id_, df_B.columns.difference(['ID', 'Interaction'])].values[0].astype(np.float32)
        # 取Interaction标签（假设两个通道标签一致）
        interaction = df_A.loc[df_A['ID'] == id_, 'Interaction'].values[0]

        # 转tensor并增加batch维度
        feat_A_tensor = torch.tensor(feat_A).unsqueeze(0)
        feat_B_tensor = torch.tensor(feat_B).unsqueeze(0)

        # 融合两个通道特征
        fused_feat = attention_layer(feat_A_tensor, feat_B_tensor).detach().numpy().squeeze()

        # 拼接ID, 特征, Interaction，注意ID是字符串，特征和标签是数值
        entry = np.concatenate([[id_], fused_feat, [interaction]])
        fused_features.append(entry)

    # 定义列名
    columns = ['ID'] + [str(i) for i in range(fused_feat.size)] + ['Interaction']

    # 创建DataFrame并保存
    fused_df = pd.DataFrame(fused_features, columns=columns)
    fused_df.to_csv(output_file, index=False)

    print(f"通道A和通道B融合特征保存至: {output_file}")


if __name__ == '__main__':
    # Luo数据集
    # 第一步：药物特征融合
    fuse_features('GNN/Drug_SMILES_Features_RGCN_Features.csv',
                  'GNN/Drug_Graph_Features_RGCN_Graph_Features.csv',
                  'GNN/Drug_Fusion_Features_RGCN.csv',
                  input_dim=64, num_heads=4)

    # 第二步：蛋白质特征融合
    fuse_features('GNN/Protein_Sequence_Features_RGCN_Features.csv',
                  'GNN/Protein_Graph_Features_RGCN_Graph_Features.csv',
                  'GNN/Protein_Fusion_Features_RGCN.csv',
                  input_dim=64, num_heads=4)

    # # 第三步：药物和蛋白质特征融合并添加标签
    final_feature_fusion('GNN/Drug_Fusion_Features_RGCN.csv',
                       'GNN/Protein_Fusion_Features_RGCN.csv',
                       'heterodata/drug_protein.csv',
                       'GNN/Fusion_features_with_labels_RGCN.csv',
                        input_dim=64, num_heads=4
    )
