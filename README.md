# gru-from-scratch
A PyTorch implementation of the Gated Recurrent Unit (GRU) by Cho et al

[Paper](https://arxiv.org/abs/1406.1078) </br>
[Shakespeare Sonnets Dataset](https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt)

Sample outputs after brief training:
```python
>>> print(sample(model, dataset, "love", 512, 1000, dev, 3))
"""love, but truly write,
and then believe me, my love is as fair
as any mother's child, though not so bressed every
but ay the barner more do sing:
when steal shame was not caunte doth all above,
that in by vies, and mine own desert'st,
thou thy breast doth tive sand in a eyes were dhing in disgrace.
therefore my mistress' eyes are raven black,
her eyes so suited, and they mourners seem still,
and in my wrong, that hates whose dear love or near.

like of the lives more than this his rone:
not tasce i soul is my heart's gainsato, would, a goed be tonte
of brood for my sake even so doth she abuse me,
suffered the best if thou shouldst depart,
leaving thee living in posterity?
be not self-will besiles and will on thee,
which from love's fire took heat perpeted
than be misted tame that i on my state,
like to the lark at break of day arising,
that dib your sudst yet the sweet sembling of thee;
in it for for the live of your feartribn skine,
with wills count bat which so dull cannot croose
but weep"""
```
