import torch
from transformers import BertTokenizer, BertModel
import os
import json
import spacy
import numpy

from os.path import join, abspath, dirname

root = join(dirname(dirname(abspath(__file__))), 'bishe')
dataset = join(root, 'multi30k-dataset/data/task1/tok')
splits=['train', 'val', 'test_2016_flickr', 'test_2017_flickr', 'test_2017_mscoco']
# splits=['test_2017_mscoco']

nlp = spacy.load('en_core_web_sm')

tokenizer = BertTokenizer.from_pretrained('./bert_weight')
bert = BertModel.from_pretrained('./bert_weight')
bert.eval()

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert = BertModel.from_pretrained('bert-base-uncased')

def getNPs(sentence):
    # nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)
    nps = []
    for chunk in doc.noun_chunks:
        nps.append({
            'phrase': chunk.text,
            'head': chunk.root.lemma_
        })
    return nps

def main():
    g_index = 0
    all_results = []
    text_embedding_matrix = []
    for split in splits:
        data = join(dataset, '{split}.lc.norm.tok.en'.format(split=split))
        results = []
        with open(data, 'r') as f:
            corpus = f.readlines()
            for index, sentence in enumerate(corpus):
                # get noun phrases
                nps = getNPs(sentence)
                # get bert embeddings
                embedding = []
                for np in nps:
                    phrase = np['phrase']
                    encoded_input = tokenizer(phrase, max_length=10,
                          add_special_tokens=True, truncation=True,
                          padding=True, return_tensors="pt")
                    # [batch_size(1), token_num, embedded_dim(768)]
                    last_hidden_layer = bert(**encoded_input)[0]
                    # [1, 768]
                    phrase_embedding = last_hidden_layer[:, 0, :]
                    text_embedding_matrix.append(last_hidden_layer[0, 0, :].detach().numpy().tolist())
                    embedding.append(g_index)
                    g_index += 1
                result = {
                    'index': index,
                    'nps': nps,
                    'embedding': embedding
                }
                results.append(result)
                all_results.append(result)
                print('dealing with {0} split, {1}/{2}: '.format(split, index, len(corpus)), nps)
         # generate np_split.json
        print('generate np_{split}.json'.format(split=split))
        with open(join(root, 'phrase/np_{split}.json').format(split=split), 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4, separators=(',', ':'))

    # generate np.json
    print('generate np.json')
    with open(join(root, 'phrase/np.json'), 'w') as f:
        json.dump(all_results, f, sort_keys=True, indent=4, separators=(',', ':'))
    # generate phrase embedding.npy
    text_embedding_matrix = numpy.array(text_embedding_matrix)
    print('text_embedding_matrix shape:', text_embedding_matrix.shape)
    print('generate phrase embedding.npy')
    numpy.save(join(root, 'phrase/phrase_embedding.npy'), text_embedding_matrix)


if __name__ == '__main__':
    main()