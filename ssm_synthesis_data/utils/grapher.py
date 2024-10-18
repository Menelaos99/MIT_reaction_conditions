import networkx as nx
import matplotlib.pyplot as plt

def generateGraph(data, G):
        
    # Add target node
    target = data["target"]
    target_material = target["material_string"]
    G.add_node(target_material, label=target_material)

    # Add precursor nodes and edges to target
    for precursor in data["precursors"]:
        precursor_material = precursor["material_string"]
        G.add_node(precursor_material, label=precursor_material)
        G.add_edge(precursor_material, target_material, label='')

        # Add elements for each precursor
        for comp in precursor["composition"]:
            for element, amount in comp["elements"].items():
                G.add_node(element, label=element)
                G.add_edge(element, precursor_material, label=amount)

    # Add elements for target
    for comp in target["composition"]:
        for element, amount in comp["elements"].items():
            G.add_node(element, label=element)
            G.add_edge(element, target_material, label=amount)
        return G
    
def generateGraphFast(data, G):
    # Add target node
    target = data["target"]
    target_material = target["material_string"]
    
    if target_material not in G:
        G.add_node(target_material, label=target_material)
    
    # Use sets to track added nodes and edges
    added_nodes = set()
    added_edges = set()
    
    # Add precursor nodes and edges to target
    for precursor in data["precursors"]:
        precursor_material = precursor["material_string"]
        
        if precursor_material not in G:
            G.add_node(precursor_material, label=precursor_material)
        edge = (precursor_material, target_material)
        if edge not in added_edges:
            G.add_edge(precursor_material, target_material, label='')
            added_edges.add(edge)
        
        # Add elements for each precursor
        for comp in precursor["composition"]:
            for element, amount in comp["elements"].items():
                if element not in added_nodes:
                    G.add_node(element, label=element)
                    added_nodes.add(element)
                edge = (element, precursor_material)
                if edge not in added_edges:
                    G.add_edge(element, precursor_material, label=amount)
                    added_edges.add(edge)
    
    # Add elements for target
    for comp in target["composition"]:
        for element, amount in comp["elements"].items():
            if element not in added_nodes:
                G.add_node(element, label=element)
                added_nodes.add(element)
            edge = (element, target_material)
            if edge not in added_edges:
                G.add_edge(element, target_material, label=amount)
                added_edges.add(edge)
    
    return G


import pandas as pd
import numpy as np
import json
from utils.classes import *

filename = r'./solid-state_dataset_20200713.json'
filedata = open(filename, mode='r').read()
jsonParse = json.loads(filedata)


# Initialize directed graph
G = nx.DiGraph()

i = 0
for item in jsonParse['reactions'][200:208]:
    G = generateGraph(item, G)
    print(i)
    i = i + 1


# Visualization
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)
labels = nx.get_node_attributes(G, 'label')
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw(G, pos, with_labels=True, labels=labels, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold', arrows=True, arrowsize=20)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Synthesis Graph")
plt.show()

