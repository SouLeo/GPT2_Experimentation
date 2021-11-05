"""
@uthor: Prakhar
"""
import os
import csv
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import warnings

warnings.filterwarnings('ignore')

class MyDataset(Dataset):
    def __init__(self, data_file_name, data_dir='.data/'):
        super().__init__()

        data_path = os.path.join(data_file_name)

        self.data_list = []
        self.end_of_text_token = " <|endoftext|> "

        with open(data_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')

            for row in csv_reader:
                data_str = f"{row[0]}"
                # data_str = f"{row[0]}: {row[1]}{self.end_of_text_token}"
                self.data_list.append(data_str)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return self.data_list[item]


def get_data_loader(data_file_name):
    dataset = MyDataset(data_file_name)
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return dataset


def choose_from_top_k_top_n(probs, k=50, p=0.8):
    ind = np.argpartition(probs, -k)[-k:]
    top_prob = probs[ind]
    top_prob = {i: top_prob[idx] for idx, i in enumerate(ind)}
    sorted_top_prob = {k: v for k, v in sorted(top_prob.items(), key=lambda item: item[1], reverse=True)}

    t = 0
    f = []
    pr = []
    for k, v in sorted_top_prob.items():
        t += v
        f.append(k)
        pr.append(v)
        if t >= p:
            break
    top_prob = pr / np.sum(pr)
    token_id = np.random.choice(f, 1, p=top_prob)

    return int(token_id)


def generate(tokenizer, model, sentences, label, device):
    with torch.no_grad():
        for idx in range(sentences):
            finished = False
            cur_ids = torch.tensor(tokenizer.encode(label)).unsqueeze(0).to('cpu')
            for i in range(100):
                outputs = model(cur_ids, labels=cur_ids)
                loss, logits = outputs[:2]

                softmax_logits = torch.softmax(logits[0, -1], dim=0)

                if i < 5:
                    n = 10
                else:
                    n = 5

                next_token_id = choose_from_top_k_top_n(softmax_logits.to('cpu').numpy())  # top-k-top-n sampling
                cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(device) * next_token_id], dim=1)

                if next_token_id in tokenizer.encode('<|endoftext|>'):
                    finished = True
                    break

            if finished:
                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = tokenizer.decode(output_list)
                f.write((output_text + '\n'))
                print(output_text)

            else:
                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = tokenizer.decode(output_list)
                print(output_text)


def load_models(model_name):
    """
    Summary:
        Loading the trained model
    """
    print('Loading Trained GPT-2 Model')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model_path = model_name
    model.load_state_dict(torch.load(model_path))
    return tokenizer, model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for inferencing Text Augmentation model')

    parser.add_argument('--model_name', default='myUMRFmodel.pt.pt', type=str, action='store', help='Name of the model file')
    parser.add_argument('--sentences', type=int, default=1, action='store', help='Number of sentences in outputs')
    parser.add_argument('--data_file', default='UMRF_valid_node_corrected.tsv', type=str, action='store', help='Name of the data file')
    args = parser.parse_args()
    DATA_FILE = args.data_file
    LOADER = get_data_loader(DATA_FILE)

    #parser.add_argument('--label', type=str, default="UMRF_valid_node_corrected.tsv", action='store', help='Label for which to produce text')


    SENTENCES = args.sentences
    MODEL_NAME = args.model_name
    # LABEL = LOADER.data_list[46]
    # LABEL = "sweep for alpha contamination under the table"
    LABELS = LOADER.data_list

    DEVICE = 'cpu'


    TOKENIZER, MODEL = load_models(MODEL_NAME)
    # model = MODEL.to(DEVICE)
    f = open('predictions.txt', 'w', encoding='utf-8')
    for valid_ex in LABELS:
        generate(TOKENIZER, MODEL, SENTENCES, valid_ex, DEVICE)
    f.close()
