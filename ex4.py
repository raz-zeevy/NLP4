import sys
import time

import nltk
# download data
# nltk.download()
from nltk.corpus import dependency_treebank
import numpy as np
import pickle

from typing import Tuple

from Chu_Liu_Edmonds_algorithm import min_spanning_arborescence_nx
from arc import Arc

NODE_TAG = 'tag'
NODE_WORD = 'lemma'
NODE_DEPS = 'deps'
POS = "POS"
WORD = "WORD"
ROOT_WORD = None
ROOT_POS = "TOP"


class MSTParser:
    def __init__(self, words, eta):
        self.eta = eta
        self.w = None
        self.v, self.p = None, None
        self.feature_mapping = {}
        self.init_data_sizes(words)

    def init_data_sizes(self, words):
        unique_words = set()
        unique_poses = set()
        for word, pos in words:
            unique_words.add(word)
            unique_poses.add(pos)
        self.v, self.p = len(unique_words), len(unique_poses)
        words_mapping = {word: i for i, word in enumerate(unique_words)}
        pos_mapping = {pos: i for i, pos in enumerate(unique_poses)}
        self.feature_mapping = {**words_mapping, **pos_mapping,
                                ROOT_WORD: self.v ** 2 + self.p ** 2}

    def get_feature_index(self, feature, feautre_type):
        head, tail = feature
        if head == ROOT_WORD or tail == ROOT_WORD or \
                head == ROOT_POS or tail == ROOT_POS:
            return self.feature_mapping[ROOT_WORD]
        if feautre_type == WORD:
            return self.feature_mapping[head] * self.v + \
                   self.feature_mapping[tail]
        if feautre_type == POS:
            return self.v ** 2 + self.feature_mapping[head] * self.p + \
                   self.feature_mapping[tail]

    def train(self, train_parsed_sentences, epochs):
        weights_sum = self.init_feature_vector()
        last_weights = self.init_feature_vector()
        gold_standart_trees = dependency_treebank.parsed_sents()
        for epoch in range(epochs):
            for i in range(len(train_parsed_sentences)):
                G = self.get_sentence_full_graph(train_parsed_sentences[i],
                                                 last_weights)
                mst = self.get_mst_from_sentence(G)
                diff = self.get_tree_vec_sum(gold_standart_trees[i]) - \
                       self.get_mst_vec_sum(mst)
                sentence_weights = last_weights + eta * diff
                weights_sum += sentence_weights
                last_weights = sentence_weights
        self.w = weights_sum / (epochs * len(train_parsed_sentences))

    def timed_train(self, train_parsed_sentences, epochs):
        weights_sum = self.init_feature_vector()
        last_weights = self.init_feature_vector()
        gold_standart_trees = dependency_treebank.parsed_sents()
        for epoch in range(epochs):
            for i in range(len(train_parsed_sentences)):
                G = self.time_it(self.get_sentence_full_graph,
                    train_parsed_sentences[i], last_weights)
                mst = self.time_it(self.get_mst_from_sentence, G)
                diff = self.get_tree_vec_sum(gold_standart_trees[i]) - \
                       self.get_mst_vec_sum(mst)
                sentence_weights = last_weights + eta * diff
                weights_sum += sentence_weights
                last_weights = sentence_weights
        self.w = self.time_it(lambda x: x / (epochs * len(
            train_parsed_sentences)), weights_sum)

    def get_sentence_full_graph(self, sentence, weights):
        graph = []
        for head in sentence.nodes.items():
            for tail in sentence.nodes.items():
                minus_weight = -1 * self.caluclate_arc_score(head[1], tail[1],
                                                             weights)
                graph.append(Arc(head[0], tail[0], minus_weight,
                                 head[1], tail[1]))
        return graph

    def predict(self, sentence) -> dict:
        G = self.get_sentence_full_graph(sentence, self.w)
        return self.get_mst_from_sentence(G)

    def eval(self, test_parsed_sentences):
        # evaluate the model accuracy
        acc_count = 0
        gold_standart_trees = dependency_treebank.parsed_sents()
        for sentence in test_parsed_sentences:
            pred_tree = self.predict(sentence)
            gold_tree = gold_standart_trees.pop(0)
            for tail_index in pred_tree:
                head_index = pred_tree[tail_index].head
                dep_itr = gold_tree.nodes[head_index][NODE_DEPS].values()
                dependency_list = list(dep_itr)[0] if dep_itr else []
                if tail_index in dependency_list:
                    acc_count += 1 / len(pred_tree)
        return acc_count / len(test_parsed_sentences)

    def get_mst_from_sentence(self, G):
        return min_spanning_arborescence_nx(G, 0)

    def get_tree_vec_sum(self, parsed_sent) -> np.ndarray:
        tree_sum_vec = self.init_feature_vector()
        mean_added_value = 1 / len(parsed_sent.nodes)
        for head_node in parsed_sent.nodes.values():
            dependency_list = list(head_node[NODE_DEPS].values())[0] if \
                head_node[NODE_DEPS].values() else []
            for tail_key in dependency_list:
                tail_node = parsed_sent.nodes[tail_key]
                # head = head_node[NODE_WORD], head_node[NODE_TAG]
                # tail = tail_node[NODE_WORD], tail_node[NODE_TAG]
                indices = self.extract_feature_indices_from_node(head_node,
                                                                 tail_node)
                tree_sum_vec[list(indices)] += mean_added_value
        return tree_sum_vec

    def caluclate_arc_score(self, head : "Node", tail : "Node", weights) -> \
            float:
        # calculate the score of the arc
        # The score is calculated by w.T @ f (feature vector)
        # since f is binary vector, we can calculate the score
        # by summing the weights of the features that are 1
        indices = self.extract_feature_indices_from_node(head, tail)
        return weights[list(indices)].sum()

    def extract_feature_indices_from_node(self, head, tail) -> (int, int):
        head_word, head_pos = head[NODE_WORD], head[NODE_TAG]
        tail_word, tail_pos = tail[NODE_WORD], tail[NODE_TAG]
        return self.get_feature_index((head_word, tail_word), WORD), \
               self.get_feature_index((head_pos, tail_pos), POS)

    def init_feature_vector(self):
        # the dtype="float32" is important to decrease the memory usage
        return np.zeros(self.v ** 2 + self.p ** 2 + 1, dtype="float32")

    def extract_feature_function(self, head: Tuple[str, str],
                                 tail: Tuple[str, str]) -> np.ndarray:
        # extract features from the dependency tree
        head_word, head_pos = head
        tail_word, tail_pos = tail
        vec = self.init_feature_vector()
        vec[self.get_feature_index((head_word, tail_word), WORD)] = 1
        vec[self.get_feature_index((head_pos, tail_pos), POS)] = 1
        return vec

    def get_mst_vec_sum(self, mst : dict):
        tree_sum_vec = self.init_feature_vector()
        mean_added_value = 1 / len(mst)
        for arc in mst.values():
            indices = self.extract_feature_indices_from_node(arc.head_node,
                                                             arc.tail_node)
            tree_sum_vec[list(indices)] += mean_added_value
        return tree_sum_vec

    @staticmethod
    def time_it(func, *args):
        start = time.time()
        ret = func(*args)
        print(func.__name__+": "+str(time.time() - start)+" seconds")
        return ret

if __name__ == '__main__':
    # "Usage: python3 ex4.py <number_of_lines>"
    global start_time
    args = sys.argv
    epochs = 2
    eta = 1
    start_time = time.time()
    parsed_sents = dependency_treebank.parsed_sents()
    parser = MSTParser(words=dependency_treebank.tagged_words(),
                       eta=eta)
    test_len = None
    if len(args) == 2:
        n = int(args[1])
        parser.train(dependency_treebank.parsed_sents()[:n], epochs=epochs)
        print("Accuracy: "+str(parser.eval(dependency_treebank.parsed_sents()[
                                           :n])))
        test_len = n
    else:
        train_portion = int((len(parsed_sents) * 0.9) // 10)
        parser.train(dependency_treebank.parsed_sents()[:train_portion], epochs=epochs)
        print("Accuracy: "+str(parser.eval(dependency_treebank.parsed_sents()[
                                           train_portion:])))
        test_len = len(parsed_sents) - train_portion
    print("on {} sentences".format(test_len))
    print(" ----------- %s seconds ------------" % (time.time() - start_time))
