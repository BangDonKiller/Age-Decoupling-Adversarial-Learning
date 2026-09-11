'''
This part is used to train the speaker model and evaluate the performances
'''

from tool.linear_decorr_training_utils import WarmupExpDecayLR
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from loss.arcface import ArcMarginProduct
from model.feature_extractor.ecapa_tdnn import ECAPA_TDNN
from tqdm import tqdm
from tool.EER import compute_eer


class ECAPAModel(nn.Module):
    def __init__(self, C , n_class, m, s):
        super(ECAPAModel, self).__init__()
        ## ECAPA-TDNN
        self.speaker_encoder = ECAPA_TDNN(C = C).cuda()
        ## Classifier
        self.speaker_loss    = ArcMarginProduct(in_features=192, out_features = n_class, m = m, s = s).cuda()

        self.optim = torch.optim.Adam(
            list(self.parameters()), 
            lr=1e-3,
            weight_decay=1e-5
        )
  
    def train_network(self, epoch, loader, optimizer=None):
        self.train()

        if optimizer is None:
            optimizer = self.optim

        index, top1, loss = 0, 0, 0
        lr = optimizer.param_groups[0]['lr']
        for num, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}", ncols=100), start=1):
            data, labels = batch[:2]
            optimizer.zero_grad()
            labels = torch.LongTensor(labels).cuda()
            speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug=True)
            nloss, prec = self.speaker_loss.forward(speaker_embedding, labels)
            nloss.backward()
            optimizer.step()

            batch_size = len(labels)
            index += batch_size
            top1 += (prec / 100.0) * batch_size
            loss += nloss.detach().cpu().numpy()

        avg_loss = loss / num
        avg_acc = top1 / index * 100.0

        return avg_loss, avg_acc
    
    def validate_network(self, epoch, loader):
        self.eval()
        index, top1, loss = 0, 0, 0
        with torch.no_grad():
            for num, batch in enumerate(tqdm(loader, desc=f"Validation Epoch {epoch+1}", ncols=100), start=1):
                data, labels = batch[:2]
                labels = torch.LongTensor(labels).cuda()
                speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug=False)
                nloss, prec = self.speaker_loss.forward(speaker_embedding, labels)

                batch_size = len(labels)
                index += batch_size
                top1 += (prec / 100.0) * batch_size
                loss += nloss.detach().cpu().numpy()

        avg_loss = loss / num
        avg_acc = top1 / index * 100.0

        return avg_loss, avg_acc        
        
    def evaluate_zero_shot(self, test_loader, device):
        """
        Returns EER computed on cosine similarity of encoder embeddings.
        """
        self.eval()

        before_scores = []
        labels = []

        with torch.no_grad():
            for is_same, id1, id2, wav1, wav2, age1, age2 in tqdm(test_loader, desc="Zero-shot Eval", leave=False):
                # wav1, wav2 expected as tensors (batch, samples)
                wav1 = wav1.to(device, non_blocking=True)
                wav2 = wav2.to(device, non_blocking=True)

                # extract speaker embeddings (evaluation, no augmentation)
                emb1 = self.speaker_encoder.forward(wav1, aug=False)
                emb2 = self.speaker_encoder.forward(wav2, aug=False)

                h1 = F.normalize(emb1, p=2, dim=1)
                h2 = F.normalize(emb2, p=2, dim=1)

                score = F.cosine_similarity(h1, h2).cpu()
                before_scores.append(score)

                labels.extend(is_same.cpu().tolist())

        final_labels = np.array(labels)
        final_before_scores = torch.cat(before_scores).numpy() if before_scores else np.array([])

        eer_before = compute_eer(final_before_scores, final_labels) if final_before_scores.size > 0 else float("nan")
        return eer_before

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model."%origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)