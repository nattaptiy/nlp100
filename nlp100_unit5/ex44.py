from lib import build_morphs,Morph,Chunk,clean_sym,chk_pos
import pydot_ng as pydot

fname = 'neko2.txt.cabocha'

def make_graph(edge_list, directed=False):

    if directed:
        graph = pydot.Dot(graph_type='digraph')

    else:
        graph = pydot.Dot(graph_type='graph')

    for edge in edge_list:

        id1 = str(edge[0][0])
        label1 = str(edge[0][1])
        id2 = str(edge[1][0])
        label2 = str(edge[1][1])

        graph.add_node(pydot.Node(id1, label='<<font face="MS Gothic">%s</font>>' % label1.rstrip("\r\n")))
        graph.add_node(pydot.Node(id2, label='<<font face="MS Gothic">%s</font>>' % label2.rstrip("\r\n")))

        graph.add_edge(pydot.Edge(id1, id2))

    return graph

for chunks in build_morphs(fname):
    edges = []
    for i, chunk in enumerate(chunks):
        if chunk.dst != -1:
            src = clean_sym(chunk)
            dst = clean_sym(chunks[chunk.dst])
            if src != '' and dst != '':
                edges.append(((i, src), (chunk.dst, dst)))
    if len(edges) > 0:
        graph = make_graph(edges, directed=True)
        graph.write_png('graph.png')