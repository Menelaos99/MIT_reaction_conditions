import time
import ujson as json
from neo4j import GraphDatabase
import matplotlib.pyplot as plt
from trade_secrets import *

class Neo4jGraph:
    def __init__(self, host, user, password):
        self.driver = GraphDatabase.driver(host, auth=(user, password))
        
    def close(self):
        self.driver.close()

    def create_graph(self, data):
        with self.driver.session() as session:
            session.write_transaction(self._create_and_link_nodes, data)

    def reset_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    @staticmethod
    def _create_and_link_nodes(tx, data):
        target = data["target"]
        target_material = target["material_formula"]
        print("Creating graph for item: ", target['material_formula'])
        # Create target node
        tx.run("MERGE (t:Material {name: $name})", name=target_material)
        is_calc = len([op for op in data['operations'] if 'calc' in op['token']])>0
        is_sint = len([op for op in data['operations'] if 'sint' in op['token']])>0
        calc_temp = -273
        sint_temp = -273    
        if is_calc:
            try:
                calc_temp = [op for op in data['operations'] if 'calc' in op['token']][0]['conditions']['heating_temperature'][0]['values'][0]
            except: calc_temp = -273
        if is_sint:
            try:
                sint_temp = [op for op in data['operations'] if 'sint' in op['token']][0]['conditions']['heating_temperature'][0]['values'][0]
            except: sint_temp = -273
        # Create precursor nodes and relationships
        for precursor in data["precursors"]:
            precursor_material = precursor["material_formula"]
            tx.run("MERGE (p:Material {name: $name, isCalc: $calc, isSint: $sint, calcTemp: $calcT, sintTemp: $sintT})", name=precursor_material, calc=is_calc, sint=is_sint, calcT=calc_temp, sintT=sint_temp)
            tx.run("""
            MATCH (p:Material {name: $precursor}), (t:Material {name: $target})
            MERGE (p)-[:PRECURSOR_TO]->(t)
            """, precursor=precursor_material, target=target_material)
            
            # Create element nodes and relationships for precursors
            for comp in precursor["composition"]:
                for element, amount in comp["elements"].items():
                    tx.run("MERGE (e:Element {name: $name})", name=element)
                    tx.run("""
                    MATCH (e:Element {name: $element}), (p:Material {name: $precursor})
                    MERGE (e)-[:PART_OF {amount: $amount}]->(p)
                    """, element=element, precursor=precursor_material, amount=amount)
        
        # Create element nodes and relationships for target
        for comp in target["composition"]:
            for element, amount in comp["elements"].items():
                tx.run("MERGE (e:Element {name: $name})", name=element)
                tx.run("""
                MATCH (e:Element {name: $element}), (t:Material {name: $target})
                MERGE (e)-[:PART_OF {amount: $amount}]->(t)
                """, element=element, target=target_material, amount=amount)

# Example usage
host = NEO4J_URI  # Replace with your actual host
user = NEO4J_USERNAME  # Replace with your actual username
password = NEO4J_PASSWORD  # Replace with your actual password

neo4j_graph = Neo4jGraph(host, user, password)
neo4j_graph.reset_database()

# Load JSON data
filename = r'./solid-state_dataset_20200713.json'
with open(filename, 'r') as file:
    jsonParse = json.load(file)

start_time = time.time()


for item in jsonParse['reactions'][:100]:  # Adjust range as needed
    print("Creating graph for item: ", item['target']['material_formula'])
    neo4j_graph.create_graph(item)

end_time = time.time()
print("Total time taken for graph generation in Neo4j: ", end_time - start_time)

neo4j_graph.close()


# Visualization using networkx and matplotlib (if needed)
# Note: Neo4j visualization can also be done within the Neo4j browser
