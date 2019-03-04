###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse
import json

import torch
from torch.autograd import Variable
from tqdm import tqdm

import data

parser = argparse.ArgumentParser(description='Generate from PyTorch AWD LSTM LM')

# Model parameters.
parser.add_argument('--corpus-path', type=str, required=True,
                    help=('Location of the data corpus. Required to '
                          'recalculate the vocab.'))
parser.add_argument('--model-path', type=str, required=True,
                    help='model checkpoint to use')
parser.add_argument('--data-path', type=str, required=True,
                    help='Path to the completion examples.')
parser.add_argument('--output-path', type=str, required=True,
                    help='Path to write the output completion examples.')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

with open(args.model_path, 'rb') as f:
    model = torch.load(f, map_location=lambda storage, loc: storage)[0]
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

corpus = data.Corpus(args.corpus_path)
initial_hidden = model.init_hidden(1)
unk_index = corpus.dictionary.word2idx["<UNK>"]


def get_next_word(previous_str, topk=10):
    split_previous_str = previous_str.split(" ")
    previous_str_indices = [corpus.dictionary.word2idx.get(word, unk_index) for word in
                            split_previous_str]
    input = Variable(torch.LongTensor(previous_str_indices).unsqueeze(-1), volatile=True)
    if args.cuda:
        input.data = input.data.cuda()
    output, _ = model(input, initial_hidden)
    word_weights = model.decoder(output)
    best_weights, best_indices = word_weights.topk(topk)
    best_words = [corpus.dictionary.idx2word[x] for x in best_indices[-1].data]
    best_weights = best_weights[-1].data.tolist()
    return {"words": best_words, "weights": best_weights}


with open(args.data_path) as data_file, open(args.output_path, "w") as output_file:
    for line in tqdm(data_file):
        instance = json.loads(line)
        next_word_output = get_next_word(" ".join(instance["prefix"]))
        output_file.write("{}\n".format(json.dumps(next_word_output)))
