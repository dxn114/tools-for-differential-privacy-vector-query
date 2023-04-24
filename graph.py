import copy
class Vertex:
    #All vertices have a unique seqno
    _seqno=0
    _neighbors = set()
    def __init__(self, seq:int) -> None:
        self._seqno=seq
        self._neighbors=set()

    #Ensure that exactly one seq is in neighbors 
    def insert_neighbor(self, seq:int)->None:
        if not seq == self._seqno:
            self._neighbors.add(seq)
    #Ensure that seq is not in neighbors 
    def remove_neighbor(self,seq:int)->None:
        self._neighbors.discard(seq)
    def neighbors(self)->set:
        return copy.deepcopy(self._neighbors)
    def degree(self)->int:
        return len(self._neighbors)

class Graph:
    _adjlist : dict[int,Vertex] = {}

    def insert_vertex(self,seq:int)->None:
        self._adjlist[seq]=Vertex(seq)

    def __init__(self,first_seq:int) -> None:
        self._adjlist={}
        self.insert_vertex(first_seq)
        
    def insert_edge(self,seq1:int,seq2:int)->None:
        if seq1 == seq2:
            return
        self._adjlist[seq1].insert_neighbor(seq2)
        self._adjlist[seq2].insert_neighbor(seq1)

    def remove_vertex(self,seq:int)->None:
        del self._adjlist[seq]
        for s in self._adjlist:
            self._adjlist[s].remove_neighbor(seq)

    def remove_edge(self,seq1:int,seq2:int)->None:
        self._adjlist[seq1].remove_neighbor(seq2)
        self._adjlist[seq2].remove_neighbor(seq1)
        
    def size(self)->int:
        return len(self._adjlist)
    
    def neighborhood(self,seq:int)->set:
        return self._adjlist[seq].neighbors()
    def edge_num(self)->int:
        num = 0
        for v in self._adjlist.values():
            num += v.degree()
        return num/2