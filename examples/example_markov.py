# Vassilis Xanthopoulos 
# Markov clustering method for ngg graph comparison 

from pyinsect.documentModel.comparators.NGramGraphSimilarity import SimilarityMarkov, SimilarityNVS
from pyinsect.documentModel.representations.DocumentNGramGraph import DocumentNGramGraph 

from scipy.spatial.distance import hamming
import markov_clustering as mc
import numpy as np
import networkx as nx
import random 

def alterSentence(sentence, k=1):
    # implement with set to avoid linear search time
    indexes = set()
    for index, ch in enumerate(sentence):
        if ch != ' ':
            indexes.add(index)

    k = min(k, len(indexes))
    for _ in range(k):
        # random choice directly on set is very slow
        # so we convert to tuple first 
        index = random.sample(tuple(indexes), 1)[0]
        indexes.remove(index)
        
        repl = random.choice("abcdefghijklmnopqrstuvwxyz")
        sentence = sentence[:index] + repl + sentence[index+1:]

    return sentence

# set the seed for our (not so) random experiments
random.seed(42)
n, Dwin = 3, 2

debug = False
if debug:
    sm = SimilarityMarkov()
    sentence1 = "abcde"
    sentence2 = "tests"

    ngg1 = DocumentNGramGraph(n, Dwin, sentence1)
    ngg2 = DocumentNGramGraph(n, Dwin, sentence2)

    print(sm(ngg1, ngg2))

    exit()

lorem_ipsum = """Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo ligula eget dolor. Aenean massa. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Donec quam felis, ultricies nec, pellentesque eu, pretium quis, sem. Nulla consequat massa quis enim. Donec pede justo, fringilla vel, aliquet nec, vulputate eget, arcu. In enim justo, rhoncus ut, imperdiet a, venenatis vitae, justo. Nullam dictum felis eu pede mollis pretium. Integer tincidunt. Cras dapibus. Vivamus elementum semper nisi. Aenean vulputate eleifend tellus. Aenean leo ligula, porttitor eu, consequat vitae, eleifend ac, enim. Aliquam lorem ante, dapibus in, viverra quis, feugiat a, tellus. Phasellus viverra nulla ut metus varius laoreet. Quisque rutrum. Aenean imperdiet. Etiam ultricies nisi vel augue. Curabitur ullamcorper ultricies nisi. Nam eget dui. Etiam rhoncus. Maecenas tempus, tellus eget condimentum rhoncus, sem quam semper libero, sit amet adipiscing sem neque sed ipsum. Nam quam nunc, blandit vel, luctus pulvinar, hendrerit id, lorem. Maecenas nec odio et ante tincidunt tempus. Donec vitae sapien ut libero venenatis faucibus. Nullam quis ante. Etiam sit amet orci eget eros faucibus tincidunt. Duis leo. Sed fringilla mauris sit amet nibh. Donec sodales sagittis magna. Sed consequat, leo eget bibendum sodales, augue velit cursus nunc"""

# take the n first sentences of lorem ipsum (gradually increase the text size)
sample_sentences = ["".join(lorem_ipsum.split('.')[:i]) for i in range(1, 20)]

repetitions = 3
alterations = [10 * x for x in range(1, 8)] #percent changes

sm = SimilarityMarkov()
snvs = SimilarityNVS()
for sentence in sample_sentences:
    print(f'Original text length: {len(sentence)}')
    ngg = DocumentNGramGraph(n, Dwin, sentence)
    
    for alt_per in alterations:
        alts = len(sentence) * alt_per // 100
        res_sm = []
        res_snvs = []
        print(f'Number of alterations: {alts} ({alt_per}%)')
        print("------------------------------------")
        for rep in range(repetitions):
            sentence_alt = alterSentence(sentence, alts)
            ngg_alt = DocumentNGramGraph(n, Dwin, sentence_alt)

            res_sm.append(sm(ngg, ngg_alt))
            res_snvs.append(snvs(ngg, ngg_alt))

        print(f"""Markov Similarity
            mean -> {np.mean(res_sm)} 
            std  -> {np.std(res_sm)}\n"""
        )
        print(f"""Normalized Value Similarity 
            mean -> {np.mean(res_snvs)} 
            std  -> {np.std(res_snvs)}\n\n"""
        )
