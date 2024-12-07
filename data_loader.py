import pandas as pd  
import torch
from torch.nn.utils.rnn import pad_sequence  
from torch.utils.data import DataLoader, Dataset
from PIL import Image  
import torchvision.transforms as transforms
import re
from collections import Counter

import re
from collections import Counter

class Vocabulary:
    def __init__(self, freq_threshold):
        self.specials = ["<pad>", "<start>", "<end>", "<unk>"]
        self.freq_threshold = freq_threshold
        self.itos = []  # Index-to-string mapping
        self.word_to_index = {}  # String-to-index mapping

    def __len__(self):
        return len(self.itos)

    def tokenize(self, text):
        return [s.lower() for s in re.split(r'\W+', text) if len(s) > 2]

    def build_vocabulary(self, sentence_dictionary):
        all_tokens = []
        for descs in sentence_dictionary:
            tokens = self.tokenize(descs)
            all_tokens.extend(tokens)
        
        word_freqs = Counter(all_tokens)
        self.index_to_word = self.specials + [word for word, count in word_freqs.items() if count >= self.freq_threshold]
        self.word_to_index = {word: idx for idx, word in enumerate(self.index_to_word)}
    def numericalize(self, text):
        
        tokenized_text = self.tokenize(text)
        encoded = []
        for token in tokenized_text:
          if token in self.word_to_index:
              encoded.append(self.word_to_index[token])
          else:
            encoded.append(self.word_to_index["<unk>"])
        return encoded

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file, sep = ",")
        with open(root_dir, 'rb') as f:
            self.image_data = pickle.load(f)
            
        self.df = self.df[self.df['image'].isin(self.image_data.keys())]
        self.df = self.df.reset_index(drop=True)


        self.imgs = self.df["image"]
        self.captions = self.df["captions"]


        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = self.image_data[img_id]

        numericalized_caption = [self.vocab.word_to_index["<start>"]]
        numericalized_caption.extend(self.vocab.numericalize(caption))
        numericalized_caption.append(self.vocab.word_to_index["<end>"])

        return img, torch.tensor(numericalized_caption)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)

        return imgs, targets


def get_loader(
    root_folder,
    annotation_file,
    transform = None,
    batch_size=32,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
):
    dataset = FlickrDataset(root_folder, annotation_file, transform=None)

    pad_idx = dataset.vocab.word_to_index["<pad>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset


if __name__ == "__main__":
    transform = None

    loader, dataset = get_loader(
        "torch_mapping.pkl", "text/captions.csv", transform=None
    )

    # for idx, (imgs, captions) in enumerate(loader):
    #     # print(imgs.shape)
    #     # print(f"image is : {imgs[0][0]}")
    #     print(captions.shape)
    #     print("captions are", captions)
    #     print("Done for this part ---------------------------------------------------------------------")
    #     break