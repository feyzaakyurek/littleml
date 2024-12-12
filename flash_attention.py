import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


class Attention(nn.Module):
    def __init__(self, hidden, seed=0):
        super().__init__()
        self.hidden = hidden
        torch.manual_seed(seed)
        self.attn = nn.Linear(hidden, 3 * hidden)  # h,3h

    def forward(self, x):
        """x: B, N, D"""
        B, N, D = x.size()
        assert D == self.hidden
        out = self.attn(x)
        q, k, v = out.split(self.hidden, dim=-1)  # B,N,D
        weights = q @ k.transpose(-2, -1)  # B,N,N
        weights = F.softmax(weights, dim=-1)
        out = weights @ v  # B,N,D
        return out


class CausalAttention(Attention):
    def forward(self, x):
        """x: B, N, D"""
        B, N, D = x.size()
        assert D == self.hidden
        q, k, v = self.attn(x).split(3, dim=-1)  # B,N,D
        weights = q @ k.transpose(-2, -1)  # B,N,N
        tril = torch.tril(torch.ones(N, N))
        weights = weights.masked_fill(tril == 0, "-inf") / (D**0.5)
        weights = F.softmax(weights, dim=-1)
        out = weights @ v  # B,N,D
        return out


class FlashAttention(nn.Module):
    def __init__(self, N, d, sram_m, seed=0):
        super().__init__()
        self.d = d  # hidden size
        self.N = N  # block size
        self.M = sram_m  # SRAM size
        # Number of tokens to attend at a time
        self.Bc = self.M // (4 * d)
        # Number of queries to consider at a time
        self.Br = min(self.M // (4 * d), d)
        # Number of blocks for queries and keys
        self.Tr, self.Tc = (N // self.Br, N // self.Bc)
        torch.manual_seed(seed)
        self.attn = nn.Linear(d, 3 * d)  # Attention layer
        # Softmax denominator
        self.l = [torch.zeros(self.Br) for _ in range(self.Tr)]
        # Max of exponents for numerical stability
        self.m = [
            torch.full((self.Br,), float("-inf")) for _ in range(self.Tr)
        ]

    def forward(self, x, device="cuda"):
        N, d = x.size()  # TODO: Assuming batch size is 1.
        assert self.d == d

        Q, K, V = self.attn(x).split(self.d, dim=-1)
        Qs = list(Q.split(self.Br, dim=0))  # each Br, d
        Ks = list(K.split(self.Bc, dim=0))  # each Bc, d
        Vs = list(V.split(self.Bc, dim=0))  # each Bc, d

        # Output of total size N, d split into blocks of size Br, d
        Os = [torch.zeros(self.Br, d) for _ in range(self.Tr)]

        for j in range(self.Tc):
            for i in range(self.Tr):
                # Compute weights.
                Sij = Qs[i] @ Ks[j].transpose(-2, -1)  # Br, Bc
                # For each query, find max weight
                mij, _ = torch.max(Sij, dim=-1)  # Br, 1
                # Softmax nominator
                Pij = torch.exp(Sij - mij)  # Br, Bc
                # Softmax denominator, one for each query
                lij = torch.sum(Pij, dim=-1)  # Br
                # Update max weight per query
                mi_new = torch.max(self.m[i], mij)  # Br
                # Update the sum so far by adding the previous max and subtracting mi_new
                li_new = torch.exp(self.m[i] - mi_new) * self.l[i]
                # Add the sum coming from this block ij
                li_new += torch.exp(mij - mi_new) * lij
                # Multiply by previous l, divide by updated l
                # also, add the new block of output ij
                Os[i] = li_new**-1 * (
                    self.l[i] * torch.exp(self.m[i] - mi_new) * Os[i]
                    + torch.exp(mij - mi_new) * Pij @ Vs[j]
                )
                self.l[i] = li_new
                self.m[i] = mi_new

        return Os

    def backward(self, x, device="cuda"):
        pass


def test_flash_attention():
    hidden = 8
    cw = 128
    sram = 256
    x = torch.randn(1, cw, hidden)
    with torch.no_grad():
        attn = Attention(hidden)
        out = attn(x)[0, :, :]

        fattn = FlashAttention(cw, hidden, sram)
        fout = torch.cat(fattn(x[0, :, :]), dim=0)
        ipdb.set_trace()

    assert torch.equal(out, fout)


if __name__ == "__main__":
    test_flash_attention()
