import re
import numpy as np
from collections import defaultdict
import torch_geometric 
import pandas as pd
import networkx as nx
import torch
from torch.utils.data import TensorDataset
import ast

##################################   PARSING   ############################################

# Complete periodic table with element positions (index starts from 1)
periodic_table = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
    'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
    'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}

# Function to parse the chemical formula
def parse_formula(formula):
    # Pattern to match elements, counts, and groups with parentheses
    #pattern = r'([A-Z][a-z]?|\([^\(\)]+\))(\d*)'
    # new pattern to accept non integer numbers
    pattern = r'([A-Z][a-z]?|\([^\(\)]+\))(\d*\.?\d*)'
    matches = re.findall(pattern, formula)
    composition = defaultdict(float)

    def add_composition(comp_dict):
        for element, count in matches:
            if count == '':
                count = 1
            else:
                count = float(count)
            if element.startswith('('):
                # Remove parentheses and parse the inner formula
                inner_formula = element[1:-1]
                inner_composition = parse_formula(inner_formula)

                for inner_element, inner_count in inner_composition.items():
                    comp_dict[inner_element] += inner_count * count
            else:
                comp_dict[element] += count

    add_composition(composition)

    return composition

def parse_prettyformula(formula):
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    composition=defaultdict(int)
    for element, count in matches:
        if count=='' or count == None:
            count=1
        composition[element]= int(count)
    return composition


# Function to create the compositional vector
def encode_compositional_vector(formula):
    composition = parse_formula(formula)
    total_atoms = sum(composition.values())
    vector = [0] * len(periodic_table)  # Create a zero vector with length equal to the number of elements

    for element, count in composition.items():
        index = periodic_table[element]
        vector[index - 1] = count / total_atoms  # Convert to fraction

    return vector


# input: list of materials
# output: list of compositional vectors
def convertFormulaToVectors(formula:list):
    output=list()
    for material in formula:
        output.append(encode_compositional_vector(material))
    return output

# concatenates precursors and targets to a 1 dimensional array
#input: list of compositional vectors
def concatenate(precursors=None, targets=None, vectorLength=None):
    # we had only one element that had len(precursors) 12 is it worth it increasing the sparsity for just one element of we just kick that element out ?
    # in the case we continue using concatenation we can add a way to check how many elements surpass a cerain threshold and e.g. we have 10 that are above maxPrecursors we just drop them
    maxPrecursors = 12
    maxTargets = 1
    vectorLength = vectorLength
    
    precursors = np.array(precursors).flatten()
    targets = np.array(targets).flatten()
    concat = np.zeros(vectorLength*(maxPrecursors+maxTargets))

    concat[:precursors.shape[0]] = precursors
    concat[-targets.shape[0]:] = targets

    return concat

#input: list of compositional vectors
# output: ML ready summed up version
def sumCompositional(precursors:list, targets:list, mean=True):
    prec=list()
    for i in range(len(precursors[0])):
        num=0
        for k in range(len(precursors)):
            num+=precursors[k][i]
        prec.append(num)

    targ=list()
    for i in range(len(targets[0])):
        num=0
        for k in range(len(targets)):
            num+=targets[k][i]
        targ.append(num)
    
    if mean:
        targ = np.array(targ)
        targ = targ/len(targets)
        
        prec = np.array(prec)
        prec = prec/len(precursors)
    
    return np.concatenate([prec,targ])


#input: lhs and rhs to make it ML ready
def transform1(leftSide:str, rightSide:str, concat=False):
    leftSide.replace("[","(")
    leftSide.replace("]",")")
    leftSide.replace("·","+")
    rightSide.replace("[","(")
    rightSide.replace("]",")")
    rightSide.replace("·","+")

    precursors=leftSide.split("+")
    precursors=list(map(str.strip, precursors))

    targets=rightSide.split("+")
    targets=list(map(str.strip, targets))

    precVectors=convertFormulaToVectors(precursors)
    targVectors=convertFormulaToVectors(targets)

    #concatenate or sum
    if concat:
        LMready=concatenate(precVectors,targVectors)
    else:
        LMready=sumCompositional(precVectors,targVectors)

    return LMready

def transform(input_str, output_str, concat=False, use_only_target=False):
    X = []
    pattern = r'\[\s*[^][]*\s*\]'
    if use_only_target:
        for output_row in output_str:
            matches = re.findall(pattern, output_row)
            matches = [ast.literal_eval(match) for match in matches]
            """if concat:
                vec=concatenate(targets=matches, vectorLength=len(matches[0]))
            else: 
                matches = torch.tensor(matches)
                vec = torch.sum(matches, dim=0)"""
            if concat:
                vec = concatenate(targets=matches, vectorLength=len(matches[0]))
            else:
                matches = torch.tensor(matches)
                vec = torch.sum(matches, dim=0).squeeze()  # Remove extra dimension if necessary

            X.append(vec)
    else:
        for input_row, output_row in zip(input_str, output_str):

            input_matches = ast.literal_eval(input_row)      
            output_matches = ast.literal_eval(output_row)

            if concat:
                vec=concatenate(precursors=input_matches, targets=output_matches, vectorLength=len(output_matches))
                #vec = torch.unsqueeze(vec, dim=0)
                vec = torch.tensor(vec)
                #print(vec.shape)
            else: 
                input_matches = torch.tensor(input_matches)
                output_matches = torch.tensor(output_matches)
                
                input_vec = torch.sum(input_matches, dim=0)
                vec = torch.add(input_vec, output_matches)/(input_matches.shape[0]+1)
                #print(vec.shape)
            X.append(vec)
    # X = torch.stack(X).float()
    # X = X.view(X.shape[0], -1)
    
    return X


###################################################################################################
def scale_node_attr(graph_list, scaler):
    for node in graph_list:
        # Ensure that node.target is at least 2D before applying the scaler
        target = node.y.reshape(-1, 1)
        node.y = torch.tensor(scaler.transform(target).reshape(-1))
    return graph_list

def get_graph(valid_inputs_calc, lhs_actual_calc, valid_target_calc, rhs_actual_calc):
    pattern = r'\[\s*[^][]*\s*\]'
    X = []
    directed = True
    for i in range(len(valid_inputs_calc)):
        if directed:
            G = nx.DiGraph()
        else: 
            G = nx.Graph()
        target = valid_target_calc[i]
        target_embd = rhs_actual_calc[i]
        # match = re.findall(pattern, target_embd)
        target_embd = ast.literal_eval(target_embd)
        target_embd = torch.tensor(target_embd)
        G.add_node(target,x=target_embd, target=True)
        
        precursors_pattern = r'\b[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*'
        precursors = valid_inputs_calc[i]
        precursors_embd = lhs_actual_calc[i]
        
        precursors = re.findall(precursors_pattern, precursors)
        
        matches = re.findall(pattern, precursors_embd)
        matches = [ast.literal_eval(match) for match in matches]
        precursors_embd = torch.tensor(matches)

        for precursor, precursor_embd in zip(precursors, precursors_embd):
            G.add_node(precursor, x=precursor_embd, target=False)
        
        target_true_nodes = []
        target_false_nodes = []

        for node, data in G.nodes(data=True):
            if data['target']:
                target_true_nodes.append(node)
            else:
                target_false_nodes.append(node)

        for source in target_false_nodes:
            for destination in target_true_nodes:
                G.add_edge(source, destination)
        data = torch_geometric.utils.from_networkx(G)
        X.append(data)
    return X

def get_data(path_df, task=None, concat=False, use_only_target=False, featurisation=None, use_graph=None):
    df = pd.read_csv(path_df, low_memory=False)
    print(f'Loaded {path_df}')
    """if task.lower()[0] == 'c':
        calc_temp = df['calcination_temp']
        y = torch.tensor(list(calc_temp))
        # calc_temp_nonna = calc_temp.dropna()
        # bool_calc_temp = calc_temp.notna()
        
        if use_graph:
            G = nx.DiGraph()
            valid_inputs_calc = list(df["valid_input_compounds"])
            valid_target_calc = list(df["target"])
            lhs_actual_calc = list(df[f'left_hand_representation_{featurisation}'])
            rhs_actual_calc = list(df[f'target_compound_representation_{featurisation}'])
            dataset = get_graph(valid_inputs_calc, lhs_actual_calc, valid_target_calc, rhs_actual_calc, G)
            for data, target in zip(dataset, y):
                data.y = target
        else:
            lhs_actual_calc = df[f'left_hand_representation_{featurisation}']
            rhs_actual_calc = df[f'target_compound_representation_{featurisation}']
            X = transform(lhs_actual_calc, rhs_actual_calc, concat=concat, use_only_target=use_only_target)
            y = y.view(-1, 1)
            dataset = TensorDataset(X, y)"""
    sint_temp = df['sintering_temp']
    y = list(sint_temp)
    # y = torch.tensor(list(sint_temp))
    # sint_temp_nonna = sint_temp.dropna()
    # bool_sint_temp = sint_temp.notna()
    
    if use_graph:
        y = torch.tensor(y)
        valid_inputs_sint = df["valid_input_compounds"].tolist()
        valid_target_sint = df["sintering_temp"].tolist()

        lhs_actual_sint = df[f'left_hand_representation_{featurisation}'].tolist()
        rhs_actual_sint = df[f'target_compound_representation_{featurisation}'].tolist()
        dataset = get_graph(valid_inputs_sint, lhs_actual_sint, valid_target_sint, rhs_actual_sint)
        for data, target in zip(dataset, y):
            data.y = target
        # print("dataset[0].x.shape[0]", dataset[0].x.shape)
    else:
        # y = y.view(-1, 1) 
        rhs_actual_sint = df[f'target_compound_representation_{featurisation}']
        lhs_actual_sint = df[f'left_hand_representation_{featurisation}']
        X = transform(lhs_actual_sint, rhs_actual_sint, concat=concat, use_only_target=use_only_target)
        X = np.array(X)
        y = np.array(y)
        y = y.reshape(-1, 1)
        dataset = [X, y]
        # dataset = TensorDataset(X, y)

    return dataset
    
def data_loading(dataset_path, task=None, concat=False, use_only_target=False, featurisation=None, use_graph=None):
    prefixes = ["train", "test", "val"]
    nb_splits = 5
    datasets = {}

    for prefix in prefixes:
        for split in range(nb_splits):
            dataset_path_x = f"{dataset_path}/{prefix}{split}.csv"
            dataset = get_data(dataset_path_x, task=task, concat=concat, use_only_target=use_only_target, featurisation=featurisation, use_graph=use_graph)     
            datasets[f"{prefix}{split}"] = dataset
    return datasets

####################################################################################################

