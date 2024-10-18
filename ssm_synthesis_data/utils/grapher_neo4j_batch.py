import time
import ujson as json
from neo4j import GraphDatabase
import matplotlib.pyplot as plt
from trade_secrets import *
import sys

class Neo4jGraph:
    def __init__(self, host, user, password):
        self.driver = GraphDatabase.driver(host, auth=(user, password))
        
    def close(self):
        self.driver.close()

    def create_graph_batch(self, batch_data):
        cypher_queries = []
        params = {}
        for idx, data in enumerate(batch_data):
            batch_cypher, batch_params = self._generate_cypher(data, idx)
            cypher_queries.extend(batch_cypher)
            params.update(batch_params)

        with self.driver.session() as session:
            session.write_transaction(self._execute_cypher_batch, cypher_queries, params)

    def reset_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    @staticmethod
    def _generate_cypher(data, idx):
        queries = []
        target = data["target"]
        target_material = target["material_formula"]
        target_var = f"target_{idx}"
        print("Preparing Cypher for item: ", target['material_formula'])

        # Create target node
        queries.append(f"MERGE ({target_var}:Material {{name: '{target_material}'}})")

        # Create precursor nodes and relationships
        for p_idx, precursor in enumerate(data["precursors"]):
            precursor_material = precursor["material_formula"]
            precursor_var = f"precursor_{idx}_{p_idx}"

            queries.append(f"MERGE ({precursor_var}:Material {{name: '{precursor_material}'}})")
            queries.append(f"""
                WITH {precursor_var}, {target_var}
                MATCH ({precursor_var}), ({target_var})
                MERGE ({precursor_var})-[:PRECURSOR_TO]->({target_var})
            """)
            
            # Create element nodes and relationships for precursors
            for c_idx, comp in enumerate(precursor["composition"]):
                for e_idx, (element, amount) in enumerate(comp["elements"].items()):
                    element_var = f"element_{idx}_{p_idx}_{c_idx}_{e_idx}"
                    
                    queries.append(f"MERGE ({element_var}:Element {{name: '{element}'}})")
                    queries.append(f"""
                        WITH {element_var}, {precursor_var}
                        MATCH ({element_var}), ({precursor_var})
                        MERGE ({element_var})-[:PART_OF {{amount: {amount}}}]->({precursor_var})
                    """)
        
        # Create element nodes and relationships for target
        for c_idx, comp in enumerate(target["composition"]):
            for e_idx, (element, amount) in enumerate(comp["elements"].items()):
                element_var = f"target_element_{idx}_{c_idx}_{e_idx}"
                
                queries.append(f"MERGE ({element_var}:Element {{name: '{element}'}})")
                queries.append(f"""
                    WITH {element_var}, {target_var}
                    MATCH ({element_var}), ({target_var})
                    MERGE ({element_var})-[:PART_OF {{amount: {amount}}}]->({target_var})
                """)

        return queries


    def create_graph_batch(self, batch_data):
        cypher_queries = []
        for idx, data in enumerate(batch_data):
            batch_cypher = self._generate_cypher(data, idx)
            cypher_queries.extend(batch_cypher)

        with self.driver.session() as session:
            session.write_transaction(self._execute_cypher_batch, cypher_queries)

    @staticmethod
    def _execute_cypher_batch(tx, queries):
        # Combine all queries into a single string
        combined_query = "\n".join(queries)

        # Calculate the size of the combined query
        query_size = sys.getsizeof(combined_query) / 1024  # Convert to KB
        
        # Print the estimated size
        print(f"Estimated query size: {query_size:.2f} KB")
        
        # Execute the combined query
        tx.run(combined_query)

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

# Define the batch size
batch_size = 1
batch = []

for item in jsonParse['reactions']:  
    batch.append(item)
    if len(batch) == batch_size:
        neo4j_graph.create_graph_batch(batch)
        batch = []

# Process any remaining items in the last batch
if batch:
    neo4j_graph.create_graph_batch(batch)

end_time = time.time()
print("Total time taken for graph generation in Neo4j: ", end_time - start_time)

neo4j_graph.close()
