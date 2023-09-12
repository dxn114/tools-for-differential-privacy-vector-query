from HGraph import HGraph
import networkx as nx,os
class HMST(HGraph):
    def build_layer(self, lc:int):
        com_graph : nx.Graph = nx.complete_graph(self.layers[lc].nodes())
        for v0,v1 in com_graph.edges():
            com_graph[v0][v1]["weight"] = self.dist(self.data[v0],v1,v0)

        self.layers[lc] = nx.minimum_spanning_tree(com_graph)

if __name__ == '__main__':
    h = HMST()
    dir_path = os.path.join(f"randvec_{h.__class__.__name__}","10^3") 
    csv_path = os.path.join(dir_path,"randvec128_10^3.csv") 
    h_path = csv_path.replace(".csv",f".{h.__class__.__name__.lower()}")
    h.build(csv_path,16)
    h.save(h_path)
    n = HMST()
    n.load(h_path)
    n.draw(dir_path)
    pass