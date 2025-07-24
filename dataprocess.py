import os
import csv
import requests
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem
import ast
from sklearn.preprocessing import OneHotEncoder
from Bio.PDB import PDBParser, MMCIFParser, NeighborSearch, Polypeptide
import matplotlib.pyplot as plt
from tqdm import tqdm

# 通道A数据预处理
# 定义最大长度和n-gram
MAX_SMILES_LENGTH = 100
MAX_PROTEIN_LENGTH = 1000
SMI_NGRAM = 3
PRT_NGRAM = 3
EMBEDDING_DIM = 128  # 嵌入的维度

# 创建文件夹，如果不存在则创建
def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
# 读取文件
def read_files(protein_file, drug_file):
    proteins = pd.read_csv(protein_file, usecols=['Protein_ID', 'Sequence'])
    drugs = pd.read_csv(drug_file, usecols=['Drug_ID', 'SMILES'])
    return proteins, drugs

# 生成 n-gram 字典
def generate_ngram_dict(sequences, ngram):
    ngram_set = set()
    for seq in sequences:
        for i in range(len(seq) - ngram + 1):
            ngram_set.add(seq[i:i + ngram])
    ngram_dict = {ngram: idx + 1 for idx, ngram in enumerate(ngram_set)}
    return ngram_dict

# 将序列转换为 n-gram 输入
def convert_to_ngram_input(sequences, ngram_dict, max_len, ngram):
    ngram_input = np.zeros((len(sequences), max_len), dtype=int)

    for i, sequence in enumerate(sequences):
        # 计算序列中 n-gram 的数量
        ngram_count = len(sequence) - ngram + 1

        # 对 n-gram 序列截断或填充
        for j in range(min(ngram_count, max_len)):
            key = sequence[j:j + ngram]
            if key in ngram_dict:
                ngram_input[i, j] = ngram_dict[key]
            else:
                # 处理未找到的 n-gram，使用0表示未知n-gram
                ngram_input[i, j] = 0

        # 如果原始 n-gram 序列超过 max_len，则进行截断
        if ngram_count > max_len:
            print(f"Warning: Sequence at index {i} exceeds max_len and is truncated.")
        # 如果原始 n-gram 序列不足 max_len，数组已经自动填充为0，无需额外处理

    return ngram_input

# 准备药物和蛋白质的嵌入
def prepare_embeddings(drug_file, protein_file):
    # 读取文件
    proteins, drugs = read_files(protein_file, drug_file)

    # 生成 n-gram 字典
    smi_dict = generate_ngram_dict(drugs['SMILES'], SMI_NGRAM)
    fas_dict = generate_ngram_dict(proteins['Sequence'], PRT_NGRAM)

    # 将序列转换为 n-gram 输入
    smi_input = convert_to_ngram_input(drugs['SMILES'], smi_dict, MAX_SMILES_LENGTH, SMI_NGRAM)
    fas_input = convert_to_ngram_input(proteins['Sequence'], fas_dict, MAX_PROTEIN_LENGTH, PRT_NGRAM)

    # 创建嵌入层
    embedding_layer_smi = nn.Embedding(num_embeddings=len(smi_dict) + 1, embedding_dim=EMBEDDING_DIM)
    embedding_layer_fas = nn.Embedding(num_embeddings=len(fas_dict) + 1, embedding_dim=EMBEDDING_DIM)

    # 将 n-gram 输入转换为嵌入
    drug_embeddings = embedding_layer_smi(torch.tensor(smi_input, dtype=torch.long)).mean(dim=1)
    protein_embeddings = embedding_layer_fas(torch.tensor(fas_input, dtype=torch.long)).mean(dim=1)

    return drugs, proteins, drug_embeddings, protein_embeddings

# 通道B数据预处理
# 从sdf文件中提取构建药物分子图的相关信息
def extract_sdf_info(sdf_file_paths):
    for sdf_file_path in sdf_file_paths:
        # 解析SDF文件
        sdf_supplier = Chem.SDMolSupplier(sdf_file_path)
        drug_atoms_data = []  # 存储每个药物的所有原子信息

        for mol in sdf_supplier:
            if mol is None:
                continue  # 如果分子对象为空，则跳过

            try:
                # 尝试获取 DRUGBANK_ID
                drugbank_id = mol.GetProp("DRUGBANK_ID")

                atom_info_list = []  # 存储当前分子的所有原子信息
                edge_info_list = []  # 存储当前分子的所有边信息

                for atom_idx in range(mol.GetNumAtoms()):
                    atom = mol.GetAtomWithIdx(atom_idx)
                    atom_info = {
                        "Atom_Type": atom.GetSymbol(),  # 原子类型
                        "Degree": atom.GetDegree(),  # 原子的度
                        "Neighboring_H": atom.GetTotalNumHs(includeNeighbors=True),  # 邻接氢原子数
                        "Formal_Charge": atom.GetFormalCharge(),  # 隐式值
                        "Is_Aromatic": atom.GetIsAromatic()  # 芳香性
                    }
                    atom_info_list.append(atom_info)

                # 遍历键（边），提取键信息
                for bond in mol.GetBonds():
                    atom1 = bond.GetBeginAtomIdx()
                    atom2 = bond.GetEndAtomIdx()
                    bond_type = str(bond.GetBondType())  # 将键类型转换为字符串
                    edge_info_list.append({
                        "Atom1": atom1,
                        "Atom2": atom2,
                        "Bond_Type": bond_type
                    })

                # 将当前分子的原子信息和边信息添加到数据列表中
                drug_atoms_data.append([drugbank_id, atom_info_list, edge_info_list])
            except Exception as e:
                print(f"Error processing molecule: {e}")
                continue  # 继续处理下一个分子

        # 获取文件名并生成输出路径
        base_name = os.path.basename(sdf_file_path)  # 提取文件名
        csv_file_name = f"drug_atoms_edges_in_{base_name.split('.')[0]}.csv"  # 生成 CSV 文件名
        output_file_path = os.path.join(os.path.dirname(sdf_file_path), csv_file_name)  # 生成输出路径

        # 将数据保存为CSV文件
        df_atoms = pd.DataFrame(drug_atoms_data, columns=["DRUGBANK_ID", "Atom_Information", "Edge_Information"])
        df_atoms.to_csv(output_file_path, index=False)
        print(f"数据已成功保存到 {output_file_path}")

# 合并两个sdf文件的药物分子图信息
def merge_csv_files(smiles_file_path, atoms_edges_file_path, output_file_path):
    # 读取drug_smiles.csv文件，获取药物编号
    smiles_df = pd.read_csv(smiles_file_path, header=None,
                            names=['DRUGBANK_ID', 'SMILES'])  # 给第一列命名为DRUGBANK_ID，第二列命名为SMILES

    # 读取drug_atoms_edges.csv文件
    edges_df = pd.read_csv(atoms_edges_file_path)  # 有列名

    # 合并两个DataFrame，使用DRUGBANK_ID作为键
    merged_df = pd.merge(smiles_df, edges_df, on='DRUGBANK_ID', how='outer')  # 使用outer合并，以保留所有药物

    # 将合并后的DataFrame保存为CSV文件
    merged_df.to_csv(output_file_path, index=False)

    print(f"合并完成，结果已保存到 {output_file_path}")

# 排序提取
def extract_rows_based_on_smiles(smiles_file_path, merged_file_path, output_file_path):
    # 读取drug_smiles.csv文件，获取药物编号和SMILES
    smiles_df = pd.read_csv(smiles_file_path, header=None)  # 无列名
    smiles_drug_ids = smiles_df.iloc[:, 0].tolist()  # 第一列药物编号列表
    smiles_values = smiles_df.iloc[:, 1].tolist()  # 第二列SMILES列表

    # 读取merged_drug_atoms_edges.csv文件
    merged_df = pd.read_csv(merged_file_path)  # 有列名

    # 初始化存储提取行的列表
    extracted_rows = []

    # 遍历drug_smiles中的药物编号
    for index, drug_id in enumerate(smiles_drug_ids):
        # 检查该药物编号是否在merged_df的DRUGBANK_ID中
        if drug_id in merged_df['DRUGBANK_ID'].values:
            # 提取该行信息
            row = merged_df[merged_df['DRUGBANK_ID'] == drug_id].copy()  # 复制行以避免SettingWithCopyWarning
            # 将SMILES添加到提取的行
            row['SMILES'] = smiles_values[index]  # 使用对应的SMILES值
            extracted_rows.append(row)  # 将该行添加到提取列表
        else:
            # 如果不存在，打印药物编号
            print(f"药物编号 {drug_id} 不存在于 merged_drug_atoms_edges.csv 中，已跳过。")

    # 将提取的行合并为一个DataFrame
    if extracted_rows:
        result_df = pd.concat(extracted_rows, ignore_index=True)
        # 保存为新的CSV文件
        result_df.to_csv(output_file_path, index=False)
        print(f"提取完成，结果已保存到 {output_file_path}")
    else:
        print("没有找到任何可提取的药物编号。")

# 建立药物的分子图
def construct_drug_graph(input_file_path, output_file_path):
    # 读取输入文件
    df = pd.read_csv(input_file_path)

    atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
                  'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K','Tl',
                  'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se','Ti', 'Zn',
                  'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni','Cd', 'In', 'Mn',
                  'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Gd', 'Unknown']

    # 创建 OneHotEncoder 对象并进行 fit
    one_hot_encoder = OneHotEncoder(sparse_output=False, categories=[atom_types])
    one_hot_encoder.fit([[atom] for atom in atom_types])

    # 初始化结果列表
    results = []

    # 遍历每个药物
    for index, row in df.iterrows():
        drug_id = row['DRUGBANK_ID']
        atom_info_list = ast.literal_eval(row['Atom_Information'])  # 将字符串转换为Python对象
        edge_info_list = ast.literal_eval(row['Edge_Information'])  # 将字符串转换为Python对象

        # 统计原子总数
        atom_count = len(atom_info_list)

        # 构建原子特征矩阵 V
        V = []
        for atom in atom_info_list:
            atom_type = atom['Atom_Type']
            degree = atom['Degree']
            neighboring_h = atom['Neighboring_H']
            formal_charge = atom['Formal_Charge']
            is_aromatic = 1 if atom['Is_Aromatic'] else 0

            # 生成独热编码的原子类型特征，确保维度统一
            atom_type_encoded = one_hot_encoder.transform([[atom_type]])[0]
            features = [int(x) for x in atom_type_encoded] + \
                       [1 if degree == i else 0 for i in range(11)] + \
                       [1 if neighboring_h == i else 0 for i in range(11)] + \
                       [1 if formal_charge == i else 0 for i in range(11)] + \
                       [is_aromatic]
            V.append(features)

        # 构建边特征矩阵 E
        E = []
        for edge in edge_info_list:
            atom1 = edge['Atom1']
            atom2 = edge['Atom2']
            bond_type = edge['Bond_Type']
            bond_type_value = {'SINGLE': 1, 'DOUBLE': 2, 'AROMATIC': 3}.get(bond_type, 0)  # 默认值为0
            E.append([atom1, atom2, bond_type_value])

        # 将原子特征和边特征转换为字符串，并确保整数格式
        atom_features_str = str(V)  # 转换为字符串
        edge_features_str = str(E)   # 转换为字符串

        # 保存结果，包括原子总数、原子特征和边特征
        results.append([drug_id, atom_count, atom_features_str, edge_features_str])  # 每个药物的信息占一行

    # 创建DataFrame并保存为CSV文件
    results_df = pd.DataFrame(results, columns=['DRUGBANK_ID', 'Atom_Count', 'Atom_Features', 'Edge_Features'])
    results_df.to_csv(output_file_path, index=False)

# 统计原子特征维度
def check_atom_features_dimensions(csv_file_path):
    # 读取生成的 CSV 文件
    df = pd.read_csv(csv_file_path)

    print("统计每行的原子特征矩阵维度：")
    for index, row in df.iterrows():
        drug_id = row['DRUGBANK_ID']
        atom_features_str = row['Atom_Features']

        # 将字符串解析为 Python 对象（列表）
        atom_features = ast.literal_eval(atom_features_str)

        # 计算维度：行数和列数
        num_rows = len(atom_features)  # 原子总数（行）
        num_cols = len(atom_features[0]) if num_rows > 0 else 0  # 每个原子的特征维度（列）

        # 打印药物的 DRUGBANK_ID 和对应的维度
        print(f"Drug ID: {drug_id}, Atom_Features Dimensions: {num_rows}x{num_cols}")

# 下载PDB文件
def download_pdb_file(pdb_id, file_format='pdb', output_folder='PDB_3D_info'):
    base_url = 'https://files.rcsb.org/download/'
    url = f'{base_url}{pdb_id}.{file_format}'
    response = requests.get(url)
    if response.status_code == 200:
        os.makedirs(output_folder, exist_ok=True)
        file_path = os.path.join(output_folder, f'{pdb_id}.{file_format}')
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f'{pdb_id}.{file_format} 下载成功，保存至 {file_path}')
        return True
    else:
        print(f'下载失败，状态码：{response.status_code}')
        return False

# 从CSV文件下载PDB文件
def download_pdbs_from_csv(csv_file_path='data/3D_PDBName.csv', output_folder='PDB_3D_info'):
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            pdb_id = row[0]  # 假设文件名在第一列
            # 尝试下载PDB文件，如果失败则下载CIF文件
            if not download_pdb_file(pdb_id, file_format='pdb', output_folder=output_folder):
                download_pdb_file(pdb_id, file_format='cif', output_folder=output_folder)
    print("所有文件下载完成。")

# 获取蛋白质的三维结构文件
def get_parser(file_path):
    if file_path.endswith(".pdb"):
        return PDBParser(QUIET=True)
    elif file_path.endswith(".cif"):
        return MMCIFParser(QUIET=True)
    else:
        raise ValueError("文件格式不支持，仅支持 .pdb 或 .cif 文件")

# 提取 CA 坐标和配体残基信息
def extract_ca_coordinates_and_ligand_residues(structure, distance_threshold=5.0):
    ca_coordinates = []
    protein_atoms = []
    ligand_atoms = []
    ligand_residues = set()

    for model in structure:
        for chain in model:
            for residue in chain:
                if Polypeptide.is_aa(residue, standard=True):  # 判断是否为标准氨基酸残基
                    if 'CA' in residue:  # 如果有 CA 原子
                        ca_atom = residue['CA']
                        ca_coordinates.append(ca_atom.get_coord())
                        protein_atoms.append(ca_atom)
                else:  # 配体分子
                    for atom in residue:
                        ligand_atoms.append(atom)

    # 使用 NeighborSearch 找到与配体邻近的残基
    ns = NeighborSearch(protein_atoms)
    for ligand_atom in ligand_atoms:
        neighbors = ns.search(ligand_atom.get_coord(), distance_threshold, level='R')
        for neighbor in neighbors:
            if Polypeptide.is_aa(neighbor, standard=True):  # 判断是否为标准氨基酸残基
                ligand_residues.add(neighbor)

    return np.array(ca_coordinates), ligand_residues

# 计算残基接触图，并标记结合口袋
def calculate_contact_map_with_pocket(ca_coordinates, ligand_residues, distance_threshold):
    n_residues = len(ca_coordinates)
    contact_map = np.zeros((n_residues, n_residues), dtype=int)

    # 使用向量化计算距离，提高效率
    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            distance = np.linalg.norm(ca_coordinates[i] - ca_coordinates[j])
            if distance < distance_threshold:
                contact_map[i, j] = 1
                contact_map[j, i] = 1

    # 标记结合口袋残基
    for residue in ligand_residues:
        try:
            protein_residue_index = residue.get_id()[1] - 1  # 获取残基索引
            if 0 <= protein_residue_index < n_residues:
                contact_map[protein_residue_index, :] = 2
                contact_map[:, protein_residue_index] = 2
        except Exception as e:
            # 如果残基索引超出范围或数据不一致，记录错误
            print(f"Error marking pocket residue: {residue}, {e}")

    return contact_map

# 生成所有蛋白质的残基接触图（带结合口袋标记）
def calculate_protein_contact_maps(input_folder, output_folder, ca_distance_threshold=8.0, ligand_distance_threshold=5.0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_list = [f for f in os.listdir(input_folder) if f.endswith(".pdb") or f.endswith(".cif")]

    for file_name in tqdm(file_list, desc="Generating Contact Maps"):
        file_path = os.path.join(input_folder, file_name)

        try:
            parser = get_parser(file_path)
            structure = parser.get_structure("protein", file_path)

            # 提取 CA 坐标和配体残基
            ca_coordinates, ligand_residues = extract_ca_coordinates_and_ligand_residues(structure, ligand_distance_threshold)

            # 计算接触图
            contact_map = calculate_contact_map_with_pocket(ca_coordinates, ligand_residues, ca_distance_threshold)

            pdb_id = file_name.split('.')[0]
            output_file = os.path.join(output_folder, f"{pdb_id}_contact_map.npy")
            np.save(output_file, contact_map)
            print(f"成功计算并保存残基接触图: {output_file}")

        except Exception as e:
            print(f"处理 {file_name} 时出错: {e}")

    print("所有残基接触图计算完成！")

def contact_map_to_adjacency_matrix(contact_map):
    """
    将接触图转换为邻接矩阵，同时保留结合口袋信息
    """
    adjacency_matrix = np.zeros_like(contact_map)
    adjacency_matrix[contact_map == 1] = 1  # 普通接触
    adjacency_matrix[contact_map == 2] = 2  # 结合口袋接触
    return adjacency_matrix

def create_adjacency_matrices(input_folder, uniprot_mapping_file, output_csv):
    """
    从输入文件夹中的接触图文件和Uniprot到PDB的映射，创建邻接矩阵并存储到CSV文件中
    """
    # 读取 Uniprot 到 PDB 的映射文件
    uniprot_mapping = pd.read_csv(uniprot_mapping_file, encoding='latin1')

    # 创建字典来存储 PDB 编号到邻接矩阵的映射
    pdb_to_adjacency = {}

    # 遍历文件夹中的接触图文件，转换为邻接矩阵并存储在字典中
    for file_name in os.listdir(input_folder):
        if file_name.endswith('_contact_map.npy'):
            pdb_id = file_name.split('_')[0][-4:]  # 提取 PDB ID 后四位
            contact_map_path = os.path.join(input_folder, file_name)
            if os.path.exists(contact_map_path):
                contact_map = np.load(contact_map_path)  # 加载接触图
                adjacency_matrix = contact_map_to_adjacency_matrix(contact_map)  # 转换为邻接矩阵
                pdb_to_adjacency[pdb_id] = adjacency_matrix  # 存储到字典中

    # 创建一个空的结果列表
    adjacency_data = []

    # 遍历 Uniprot 到 PDB 的映射文件，确保每个 Uniprot_ID 对应一个邻接矩阵
    for index, row in uniprot_mapping.iterrows():
        uniprot_id = row['Protein_ID']  # Uniprot编号
        pdb_id = str(row['To'])[-4:]  # PDB编号的后四位

        if pdb_id in pdb_to_adjacency:
            # 获取邻接矩阵
            adjacency_matrix = pdb_to_adjacency[pdb_id]

            # 保存到结果列表（一个Uniprot编号对应一个记录）
            adjacency_data.append((uniprot_id, adjacency_matrix.tolist()))
        else:
            # 如果没有找到对应的邻接矩阵，跳过
            print(f"未找到PDB编号 {pdb_id} 对应的邻接矩阵。")

    # 将结果保存为CSV文件
    result_df = pd.DataFrame(adjacency_data, columns=['Protein_ID', 'Adjacency_Matrix'])
    result_df.to_csv(output_csv, index=False)
    print(f"邻接矩阵已保存到文件：{output_csv}")

if __name__ == '__main__':
    # 通道B药物分子图生成
    # sdf提取信息
    sdf_file_paths = [
        'data/3D_structures.sdf',  # SDF 文件路径
        'data/approved_drugs.sdf'  # SDF 文件路径
    ]
    extract_sdf_info(sdf_file_paths)
    
    # 合并
    smiles_file_path = 'data/drug_atoms_edges_in_3D_structures.csv'
    atoms_edges_file_path = 'data/drug_atoms_edges_in_approved_drugs.csv'
    output_file_path = 'data/merged_drug_atoms_edges.csv'  # 输出合并后的文件路径
    merge_csv_files(smiles_file_path, atoms_edges_file_path, output_file_path)
    
    # 提取
    smiles_file_path = 'data/drug_smiles.csv'  # drug_smiles.csv的路径
    merged_file_path = 'data/merged_drug_atoms_edges.csv'  # merged_drug_atoms_edges.csv的路径
    output_file_path = 'data/drug_graph_data.csv'  # 输出提取后的文件路径
    extract_rows_based_on_smiles(smiles_file_path, merged_file_path, output_file_path)
    
    # 构建药物分子图
    input_file_path = 'data/drug_graph_data.csv'  # 输入文件路径
    output_file_path = 'data/drug_graph.csv'  # 输出文件路径
    construct_drug_graph(input_file_path, output_file_path)

    # 通道B构建蛋白质残基接触图
    # 计算残基接触图
    calculate_protein_contact_maps('PDB_3D_info', 'Protein_Contact_Maps', ca_distance_threshold=8.0,
                                   ligand_distance_threshold=5.0)
    create_adjacency_matrices('Protein_Contact_Maps', 'data/Uniprot_PDB.csv', 'data/protein_graph.csv')

