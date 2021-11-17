"""
   NGramCachedGraphComparator.py

 An n-gram graph similarity class
 that calculates a set of ngram graph
 similarity measures implementing
 basic similarity extraction functions.

 @author ysig
 Created on May 24, 2017, 3:56 PM
"""

from functools import reduce

from pyinsect.documentModel.comparators.Operator import BinaryOperator


# a general similarity class
# that acts as a pseudo-interface
# defining the basic class methods
class Similarity(BinaryOperator):
    def __init__(self, commutative=True, distributional=False):
        self._commutative = commutative
        self._distributional = distributional

    # given two ngram graphs
    # returns the given similarity as double
    def getSimilarityDouble(self, ngg1, ngg2):
        return 0.0

    # given two ngram graphs
    # returns some midway extracted similarity components
    # as a dictionary between of sting keys (similarity-name)
    # and double values
    def getSimilarityComponents(self, ngg1, ngg2):
        return {"SS": 0, "VS": 0, "NVS": 0}

    # from the similarity components extracts
    # what she wants for the given class
    def getSimilarityFromComponents(self, Dict):
        return 0.0

    def apply(self, *args, **kwargs):
        return self.getSimilarityDouble(*args)

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)


class SimilaritySS(Similarity):

    # given two ngram graphs
    # returns the SS-similarity as double
    def getSimilarityDouble(self, ngg1, ngg2):
        # WRONG
        # return (min(ngg1.minW(),ngg2.minW())*1.0)/max(ngg1.maxW(),ngg2.maxW())
        y = max(ngg1.number_of_edges(), ngg2.number_of_edges())
        if y == 0:  # If both graphs are zero sized
            return 0.0  # return zero

        return (min(ngg1.number_of_edges(), ngg2.number_of_edges()) * 1.0) / max(
            ngg1.number_of_edges(), ngg2.number_of_edges()
        )

    # given two ngram graphs
    # returns the SS-similarity
    # components on a dictionary
    def getSimilarityComponents(self, ngg1, ngg2):
        return {"SS": (self.getSimilarityDouble(ngg1, ngg2))}

    # given similarity components
    # extracts the SS measure
    # if existent and returns it (as double)
    def getSimilarityFromComponents(self, Dict):
        if "SS" in Dict:
            return Dict["SS"]
        else:
            return 0.0


class SimilarityVS(Similarity):

    # given two ngram graphs
    # returns the VS-similarity as double
    def getSimilarityDouble(self, ngg1, ngg2):
        s = 0.0
        g1 = ngg1.getGraph()
        g2 = ngg2.getGraph()
        ne1 = g1.number_of_edges()
        ne2 = g2.number_of_edges()

        if ne1 == ne2 == 0:
            return 1.0

        if ne1 > ne2:
            t = g2
            g2 = g1
            g1 = t
        edges2 = set(g2.edges())  # Use set to speed up finding
        for (u, v, d) in g1.edges(data=True):
            if (u, v) in edges2:
                dp = g2.get_edge_data(u, v)
                s += min(d["weight"], dp["weight"]) / max(d["weight"], dp["weight"])
        return s / max(g1.number_of_edges(), g2.number_of_edges())

    # given two ngram graphs
    # returns the VS-similarity
    # components on a dictionary
    def getSimilarityComponents(self, ngg1, ngg2):
        return {"VS": self.getSimilarityDouble(ngg1, ngg2)}

    # given similarity components
    # extracts the SS measure
    # if existent and returns it (as double)
    def getSimilarityFromComponents(self, Dict):
        if "VS" in Dict:
            return Dict["VS"]
        else:
            return 0.0


class SimilarityNVS(Similarity):

    # given two ngram graphs
    # returns the NVS-similarity as double
    def getSimilarityDouble(self, ngg1, ngg2):
        SS = SimilaritySS()
        VS = SimilarityVS()
        return (VS.getSimilarityDouble(ngg1, ngg2) * 1.0) / SS.getSimilarityDouble(
            ngg1, ngg2
        )

    # given two ngram graphs
    # returns the NVS-similarity
    # components e.g. SS and VS
    # on a dictionary
    def getSimilarityComponents(self, ngg1, ngg2):
        SS = SimilaritySS()
        VS = SimilarityVS()
        return {
            "SS": SS.getSimilarityDouble(ngg1, ngg2),
            "VS": VS.getSimilarityDouble(ngg1, ngg2),
        }

    # given a dictionary containing
    # SS similarity and VS similarity
    # extracts NVS if SS is not 0
    def getSimilarityFromComponents(self, Dict):
        if ("SS" in Dict and "VS" in Dict) and (str(Dict["SS"]) != "0.0"):
            return (Dict["VS"] * 1.0) / Dict["SS"]
        else:
            return 0.0


class SimilarityHPG(Similarity):
    """A custom `Similarity` metric tailored to the complexities
    of Hierarchical Proximity Graphs (HPG - `DocumentNGramHGraph`).

    Given two HPGs, the `Value Similarity` of every sub-graph pair is computed,
    on a pair level basis, and the weighted mean of among all levels is considered
    the HPGs Value Similarity.
    """

    def __init__(
        self, per_level_similarity_metric_type, commutative=True, distributional=False
    ):
        super().__init__(commutative=commutative, distributional=distributional)

        self._per_level_similarity_metric = per_level_similarity_metric_type(
            commutative=commutative, distributional=distributional
        )

    def getSimilarityDouble(self, document_n_gram_h_graph1, document_n_gram_h_graph2):
        if not document_n_gram_h_graph1 and not document_n_gram_h_graph2:
            return 1

        if not document_n_gram_h_graph1 or not document_n_gram_h_graph2:
            return 0

        similarity = 0

        for level, (current_1, current_2) in enumerate(
            zip(document_n_gram_h_graph1, document_n_gram_h_graph2), start=1
        ):
            current_lvl_similarity = (
                self._per_level_similarity_metric.getSimilarityDouble(
                    current_1, current_2    
                )
            )

            similarity += level * current_lvl_similarity

        return similarity / reduce(lambda x, y: x + y, range(1, level + 1))


class SimilarityMarkov(Similarity):

    # given two ngram graphs
    # returns their similarity using markov techniques as double
    def getSimilarityDouble(self, ngg1, ngg2):

        import markov_clustering as mc
        import networkx as nx
        from scipy.spatial.distance import hamming

        g1 = ngg1.getGraph()
        g2 = ngg2.getGraph()
        n1 = g1.nodes()
        n2 = g2.nodes()

        if n1 == 0 or n2 == 0:
            return 1.0

        print(n1)
        print(n2)

        # remove duplicates and order the union of nodes of the 2 graphs, 
        # all in one ugly unmaintainable line of python code
        ordered_nodes = sorted(list(
            set(n1).union(
                set(n2)
            )
        ))

        print(ordered_nodes)

        g1.add_nodes_from(map(tuple, n2))
        g2.add_nodes_from(map(tuple, n1))

        # force an ordering on the nodes with second argument
        A1 = nx.linalg.graphmatrix.adjacency_matrix(g1, ordered_nodes)
        A2 = nx.linalg.graphmatrix.adjacency_matrix(g2, ordered_nodes)

        result1 = mc.run_mcl(A1).todense().flatten()
        result2 = mc.run_mcl(A2).todense().flatten()

        return hamming(result1, result2)


    # given two ngram graphs
    # returns the similarity
    # components on a dictionary
    def getSimilarityComponents(self, ngg1, ngg2):
        return {"Markov": self.getSimilarityDouble(ngg1, ngg2)}

    # given similarity components
    # extracts the SS measure
    # if existent and returns it as double
    def getSimilarityFromComponents(self, Dict):
        if "Markov" in Dict:
            return Dict["Markov"]
        else:
            return 0.0
