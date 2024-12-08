import torch
import torch.nn as nn
import torchvision.models as models

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.sq = nn.Linear(4096, embed_size)
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        features = self.sq(features.squeeze(1))
        # print(f"Captions before embedding: {captions}")
        # print("size of features is :",features.shape)
    
        assert captions.max() < self.embed.num_embeddings, "Caption index out of range!"
        embeddings = self.dropout(self.embed(captions))
        
        # print(f"Captions after embedding: {embeddings}")
        # print("size of caption embedding is :",embeddings.shape)
        # print("size of unsueesed features is :", features.unsqueeze(0).shape)
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
        
    def caption_image(self, feature, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():

            feature = self.sq(feature.squeeze(1))
            x = feature.unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.lstm(x, states)
                output = self.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.embed(predicted).unsqueeze(0)

                if vocabulary.index_to_word[predicted.item()] == "<end>":
                    break

        return [vocabulary.index_to_word[idx] for idx in result_caption]