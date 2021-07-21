class SelfAttention(nn.Module):
  """
  Canonical implementation of multi-head self attention.
  """

  def __init__(self, emb, heads=8, mask=False):
    super().__init__()

    # Conferm the emb / heads has no remainder
    assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

    self.emb, self.heads, self.mask = emb, heads, mask

    # - We will break the embedding into `heads` chunks and feed each to a different attention head

    self.KeysWeights = nn.Linear(emb, emb, bias=False)
    self.QueriesWeights = nn.Linear(emb, emb, bias=False)
    self.ValuesWeights  = nn.Linear(emb, emb, bias=False)

    self.UnifyHeadsWeights = nn.Linear(emb, emb)

  def forward(self, x):
    b, t, e = x.size()
    h = self.heads
    assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

    s = e // h

    keys    = self.KeysWeights(x)
    queries = self.QueriesWeights(x)
    values  = self.ValuesWeights(x)

    keys    = keys.view(b, t, h, s)
    queries = queries.view(b, t, h, s)
    values  = values.view(b, t, h, s)

    # -- We first compute the k/q/v's on the whole embedding vectors, and then split into the different heads.
    #    See the following video for an explanation: https://youtu.be/KmAISyVvE1Y

    # Compute scaled dot-product self-attention

    # - fold heads into the batch dimension
    keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
    queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
    values = values.transpose(1, 2).contiguous().view(b * h, t, s)

    queries = queries / (e ** (1/4))
    keys    = keys / (e ** (1/4))
    # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
    #   This should be more memory efficient

    # - get dot product of queries and keys, and scale
    dot = torch.bmm(queries, keys.transpose(1, 2))

    assert dot.size() == (b * h, t, t)

    if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
        mask_(dot, maskval=float('-inf'), mask_diagonal=False)

    dot = F.softmax(dot, dim=2)
    # - dot now has row-wise self-attention probabilities

    # apply the self attention to the values
    out = torch.bmm(dot, values).view(b, h, t, s)

    # swap h, t back, unify heads
    out = out.transpose(1, 2).contiguous().view(b, t, s * h)

    return self.unifyheads(out)