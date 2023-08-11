import torch
from torch.utils.data import Dataset
from utils import one_hot
from tqdm import tqdm

class SequenceDataset(Dataset):
    def __init__(self, txt_path: str, seq_length: int = 100) -> None:
        f = open(txt_path, encoding="utf-8")
        self.seq_length = seq_length
        self.corpus = "".join(f.readlines()).lower()
        self.vocab = sorted(list(set(self.corpus)))
        self.vocab_to_idx = {i: idx for (idx, i) in enumerate(self.vocab)}
        self.idx_to_vocab = {v: k for k, v in self.vocab_to_idx.items()}
        self.corpus_encoded = torch.zeros(len(self.corpus), len(self.vocab))

        for i, char in tqdm(enumerate(self.corpus), total=len(self.corpus), position=0, leave=True):
            self.corpus_encoded[i] = one_hot(self.vocab_to_idx[char], len(self.vocab))

        self.xvals = self.corpus_encoded[:-1]
        self.yvals = self.corpus_encoded[1:]

        extra_vals = len(self.xvals) % seq_length

        self.xvals = self.xvals[:-extra_vals].view(len(self.xvals) // seq_length, seq_length, len(self.vocab))
        self.yvals = self.yvals[:-extra_vals].view(len(self.yvals) // seq_length, seq_length, len(self.vocab))


    def __len__(self) -> int:
        return self.xvals.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.xvals[idx], self.yvals[idx]