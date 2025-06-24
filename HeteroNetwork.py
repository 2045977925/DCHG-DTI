import pandas as pd
import networkx as nx
import ast
from tqdm import tqdm

def build_heterogeneous_network(drug_features_file, protein_features_file, drug_similarity_file, protein_similarity_file, interaction_file, drug_id_col, protein_id_col, features_col):
    # 读取药物和蛋白质特征
    drug_features = pd.read_csv(drug_features_file)
    protein_features = pd.read_csv(protein_features_file)

    # 获取药物和蛋白质的 ID 列表
    drug_ids = set(drug_features[drug_id_col])
    protein_ids = set(protein_features[protein_id_col])

    # 读取相似度矩阵和相互作用矩阵
    drug_similarity = pd.read_csv(drug_similarity_file, index_col=0)
    protein_similarity = pd.read_csv(protein_similarity_file, index_col=0)
    drug_protein_interaction = pd.read_csv(interaction_file, index_col=0)

    # 创建一个异构网络
    G = nx.Graph()

    # 添加药物节点及其特征
    for _, row in tqdm(drug_features.iterrows(), total=drug_features.shape[0], desc="Adding drug nodes"):
        drug_id = row[drug_id_col]
        try:
            features = ast.literal_eval(row[features_col])
            G.add_node(drug_id, type='drug', features=features)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing features for drug {drug_id}: {e}")

    # 添加蛋白质节点及其特征
    for _, row in tqdm(protein_features.iterrows(), total=protein_features.shape[0], desc="Adding protein nodes"):
        protein_id = row[protein_id_col]
        try:
            features = ast.literal_eval(row[features_col])
            G.add_node(protein_id, type='protein', features=features)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing features for protein {protein_id}: {e}")

    # 添加药物之间的相似性边 (只包括有特征的药物)
    for drug1 in tqdm(drug_ids, desc="Adding drug similarity edges"):
        for drug2 in drug_ids:
            if drug1 != drug2 and drug_similarity.loc[drug1, drug2] > 0:  # 根据需要调整阈值
                G.add_edge(drug1, drug2, type='drug_similarity', weight=drug_similarity.loc[drug1, drug2])

    # 添加蛋白质之间的相似性边 (只包括有特征的蛋白质)
    for protein1 in tqdm(protein_ids, desc="Adding protein similarity edges"):
        for protein2 in protein_ids:
            if protein1 != protein2 and protein_similarity.loc[protein1, protein2] > 0:  # 根据需要调整阈值
                G.add_edge(protein1, protein2, type='protein_similarity', weight=protein_similarity.loc[protein1, protein2])

    # 添加药物-蛋白质相互作用边 (只包括有特征的药物和蛋白质)
    for drug in tqdm(drug_ids, desc="Adding drug-protein interaction edges"):
        for protein in protein_ids:
            if drug_protein_interaction.loc[drug, protein] == 1:
                G.add_edge(drug, protein, type='drug_protein_interaction')

    return G

