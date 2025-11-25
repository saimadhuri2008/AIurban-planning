from pyvis.network import Network
import networkx as nx

graphml_path = r"C:\AIurban-planning\data\processed\master\bengaluru_urban_graph.graphml"

G = nx.read_graphml(graphml_path)

net = Network(height="1000px", width="100%", notebook=False)
net.from_nx(G)
net.show("bengaluru_graph.html")
