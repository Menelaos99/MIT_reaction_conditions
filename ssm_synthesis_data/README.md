# Solid State Material Synthesis Data Processor

## Data Preview
Based on the Neo4j Graph method (mentioned below), a web-interface is hosted at [a private compute server](http://ssm.mit.de.soanpapdi.cloud/browser). For credentials please contact Kesava. The DB contains around 750 materials for testing and superficial analyses. Follow through issues for known limitations.

## Setup Instructions

1. **Download and Extract Dataset**: Download the dataset zip file and extract the JSON file to the root of the folder.
2. **Install Dependencies**: Use the provided `requirements.txt` file to install necessary packages. The key dependencies are Pandas, Numpy, Networkx, and Matplotlib. You can install them using pip:

```bash
pip install pandas numpy networkx matplotlib
```
3. Create a file named `trade_secrets.py` and fill the contents with following:

```python
NEO4J_PASSWORD = '<Password for Neo4j Dashboard>'
NEO4J_USERNAME = '<Username for Neo4j Dashboard>'
NEO4J_URI = '<Hostname for Neo4j Dashboard>'
```

## Ingesting the Dataset

To ingest the entire dataset as native Python classes, use the helper class provided in `classes.py`. Here's an example to get all the reactions:

```python
import pandas as pd
import numpy as np
import json
from classes import from_dict, ReactionEntry

filename = 'solid-state_dataset_20200713.json'

# Read the JSON file
with open(filename, 'r') as file:
    json_data = json.load(file)

# Parse the JSON data into a list of ReactionEntry objects
reactions = [from_dict(reaction, ReactionEntry) for reaction in json_data['reactions']]
```

## Graphing the Dataset

There are two ways to visualize the dataset: with or without Neo4j.

### Visualization without Neo4j

To visualize the dataset without Neo4j, run `grapher.py`. This module parses the dataset into the [Graph V2 JSON Specification](https://github.com/jsongraph/json-graph-specification) and creates a graph using NetworkX. Although NetworkX is best suited for creating Directed Acyclic Graphs (DAGs), it can still be used for this purpose. The graph's general output can be interpreted as follows:

- **Central Nodes**: Nodes in the center are used extremely frequently.
- **Target Elements**: All nodes, except root nodes, are target elements. When two target elements (A and B) form another target (C), A and B are defined as precursors in the dataset.

Conforming to the Graph V2 standards facilitates model transformations via ONNX and toolkits like TAO for one-to-one validation streams. However, this requirement should be verified based on your specific needs.

To run the visualization, execute:

```bash
python grapher.py
```

### Neo4j Graph Visualization

This project demonstrates how to set up Neo4j using Docker, connect to it, and interact with the graph database using Cypher queries. 

#### Step 1: Install Docker

**Install Docker:**

- **Windows/Mac**: Download and install Docker Desktop from Docker's official website.
- **Linux**: Follow the installation instructions specific to your distribution on Docker's official installation guide.

**Verify Docker Installation:**

Open a terminal or command prompt and run:

```bash
docker --version
```

You should see the version of Docker installed, indicating that Docker is correctly installed.

#### Step 2: Run Neo4j Using Docker

**Pull the Neo4j Docker Image:**

```bash
docker pull neo4j:latest
```

**Run the Neo4j Container:**

```bash
docker run -d \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest
```

**Verify the Neo4j Container is Running:**

```bash
docker ps
```

You should see an entry for the Neo4j container with its status as "Up".

#### Step 3: Access Neo4j Browser

**Open the Neo4j Browser:**

Open a web browser and go to [http://localhost:7474](http://localhost:7474).

**Log In:**

Use `neo4j` as the username and `password` as the password (or the password you set).

To visualize the dataset using Neo4j, run `grapher_neo4j.py`. This script will connect to the Neo4j instance and populate it with data from the dataset.

To run the Neo4j visualization, execute:

```bash
python grapher_neo4j.py
```

### Interacting with the Graph Database

Once you are logged into the Neo4j Browser, you can use Cypher, Neo4j’s query language, to interact with the graph database. Below are some common Cypher commands with explanations.

#### Key Cypher Queries

1. **Retrieve All Materials**

This query retrieves all nodes labeled as Material.

```cypher
MATCH (m:Material) RETURN m
```

2. **Retrieve All Elements**

This query retrieves all nodes labeled as Element.

```cypher
MATCH (e:Element) RETURN e
```

3. **Retrieve All Precursor Relationships**

This query retrieves all PRECURSOR_TO relationships between Material nodes.

```cypher
MATCH (p:Material)-[:PRECURSOR_TO]->(t:Material) RETURN p, t
```

4. **Retrieve Elements and Their Amounts for a Specific Material**

This query retrieves all elements and their amounts for a specific material, such as Li4Ti5O12.

```cypher
MATCH (e:Element)-[r:PART_OF]->(m:Material {name: 'Li4Ti5O12'}) RETURN e, r, m
```

5. **Find All Precursors and Their Elements for a Specific Target Material**

This query finds all precursor materials and their constituent elements for a specific target material.

```cypher
MATCH (p:Material)-[:PRECURSOR_TO]->(t:Material {name: 'Li4Ti5O12'})
MATCH (e:Element)-[r:PART_OF]->(p)
RETURN p, e, r
```

6. **Retrieve the Entire Graph**

This query retrieves all nodes and relationships in the graph.

```cypher
MATCH (n)-[r]->(m) RETURN n, r, m
```

## Experimental Flat Structure Parser

The `flat_structure_parser.ipynb` notebook contains an experimental code set that flattens the dataset structure based on maximum possibilities. For instance, a reaction has a maximum of 10 reactants (LHS width) and 5 products (RHS width). This structured approach is advantageous for Feed-forward Linear and Cross-linear Neural Networks (FCNN/Dense) but may increase iteration counts. Using Stochastic Gradient Descent (SGD) can enhance performance.

### Key Points:

- **Empty Values**: All empty values in the row are considered as having null enthalpy (in weight and bias terms, not thermodynamic), reducing the need for many hidden layers.
- **SPP Networks**: This approach is also effective for creating SPP networks in conjunction with Encoders/Decoders.
- **Element Ordering**: Elements are ordered in a non-stochastic manner, leading to higher hit probabilities and allowing integration with `periodictable`. This essentially provides new data insights.
