#!/usr/bin/env python
import argparse
import re
import operator
from dl4nlp.word2vec import skip_gram_cost_gradient
from dl4nlp.stochastic_gradient_descent import *
from dl4nlp.gradient_descent import gradient_descent


def preprocess(lines, dictionary):
    """
    Convert lines to word ids
    :param lines: iterable of strings
    :param dictionary: maps word to index of word vector
    :return: iterable of integer
    """
    tokenizer = re.compile('[ ,.?!]')

    for line in lines:
        line = line.rstrip('\n')
        line = line.lower()
        words = tokenizer.split(line)

        indices = []
        for word in words:
            if word not in dictionary:
                dictionary[word] = len(dictionary)
            index = dictionary[word]
            indices.append(index)

        yield indices


def load_data(data, context_size):
    """
    load raw data into input and output vectors of indices
    :param data: file pointer of text file
    :param context_size: integer of context size
    :return: inputs and outputs vectors
    """
    inputs = []
    outputs = []

    for sentence in data:
        for i in range(len(sentence)):
            output_row = []
            input = sentence[i]

            for j in range(-context_size, context_size + 1):
                if j == 0 or i + j < 0 or i + j >= len(sentence):
                    continue

                output = sentence[i + j]
                output_row.append(output)

            inputs.append(input)
            outputs.append(output_row)

    inputs = np.array(inputs).T
    outputs = np.array(outputs).T

    return inputs, outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=argparse.FileType())
    parser.add_argument('output_file', type=argparse.FileType('w'))
    parser.add_argument('vector_size', type=int)
    parser.add_argument('context_size', type=int)
    args = parser.parse_args()

    dictionary = {}
    data = preprocess(args.input_file, dictionary)
    inputs, outputs = load_data(data, args.context_size)

    cost_gradient = bind_cost_gradient(skip_gram_cost_gradient, inputs, outputs, sampler=get_stochastic_sampler(100))
    initial_parameters = np.random.normal(size=(2, len(dictionary), args.vector_size))
    parameters, cost_history = gradient_descent(cost_gradient, initial_parameters, 10)
    input_vectors, output_vectors = parameters
    word_vectors = input_vectors + output_vectors
    words = [word for word, index in sorted(dictionary.items(), key=operator.itemgetter(1))]

    for word, vector in zip(words, word_vectors):
        print(word, ' '.join(str(element) for element in vector), file=args.output_file)

if __name__ == '__main__':
    main()
