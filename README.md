<img align="right" width="300" height="300" src="https://user-images.githubusercontent.com/39880630/119734484-62a27180-be7b-11eb-8ef2-1f63a4209345.png">

## erdospy

Fast sampling of the `G(n, m)`-model of [Erdős–Rényi random graphs](https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model).

The time and space complexity of the algorithm is in `O(m*samples)`. The result can be returned either as an integer `np.ndarray` of size `(2, m, samples)` or a `list`of integer `sp.sparse.coo_matrix`.


#### Installation

`erdospy` can be installed directly from source with:
```sh
pip install git+https://github.com/NiMlr/erdospy
```

Test your installation via:
```python
import erdospy
erdospy.testall()
```


#### Empirical performance

Thanks to the powerful implementation of [`sklearn.utils.random.sample_without_replacement`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_random.pyx#L223),
it turns out that the method in many cases is not only faster than that of other packages
```python
from erdospy import sample_erdos_renyi_gnm
import networkx as nx

n = 50
m = 600
samples = 2000

%timeit sample_erdos_renyi_gnm(n, m, samples, return_as="edge_array")
# 116 ms ± 1.37 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
%timeit sample_erdos_renyi_gnm(n, m, samples, return_as="adjacency_matrix")
# 359 ms ± 15.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
%timeit [nx.gnm_random_graph(n, m) for _ in range(samples)]
# 4.66 s ± 142 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```
but it opens up the use-case `n >> m`

```python
n = 1_000_000
m = 600
samples = 1

%timeit sample_erdos_renyi_gnm(n, m, samples, return_as="adjacency_matrix")
# 3.6 ms ± 196 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
%timeit [nx.gnm_random_graph(n, m) for _ in range(samples)]
# 931 ms ± 16.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# running nx.gnm_random_graph for even larger n will soon break down
```


#### Implementation details

The **simple** algorithm is based on the fast sampling of `m` edge indices in `0, 1, ..., n*(n-1)//2` using [`sklearn.utils.random.sample_without_replacement`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_random.pyx#L223) and the subsequent sample-wise constant time mapping onto the indices of the adjacency matrix `A_{n, ij} -> (i, j)`, where the edge indices are associated with the corresponding matrix entry, due to undirectedness the cases `j < i` are sufficient, and
<br>
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=A_n&space;=&space;\begin{pmatrix}&space;*&space;&&space;*\phantom{*}&space;&&space;\phantom{*}*&space;&&space;\phantom{*}*\phantom{*}&space;&&space;*\phantom{*}&space;&&space;*\\&space;0&space;&&space;*\phantom{*}&space;&&space;\phantom{*}*&space;&&space;\phantom{*}*\phantom{*}&space;&&space;*\phantom{*}&space;&&space;*\\&space;1&space;&&space;2\phantom{*}&space;&&space;\phantom{*}*&space;&&space;\phantom{*}*\phantom{*}&space;&&space;*\phantom{*}&space;&&space;*\\&space;3&space;&&space;4\phantom{*}&space;&&space;\phantom{*}5&space;&&space;\phantom{*}*\phantom{*}&space;&&space;*\phantom{*}&space;&&space;*\\&space;\vdots&space;&&space;&&space;&&space;\ddots&space;&&space;\ddots&space;&\vdots\vspace*{2mm}\\&space;\tfrac{n(n-3)}{2}&space;&plus;2&space;&&space;&\phantom{*}\dots&space;&&space;&&space;\tfrac{n(n-1)}{2}\phantom{*}&space;&&space;*&space;\end{pmatrix}&space;\,." target="_blank"><img src="https://latex.codecogs.com/gif.latex?A_n&space;=&space;\begin{pmatrix}&space;*&space;&&space;*\phantom{*}&space;&&space;\phantom{*}*&space;&&space;\phantom{*}*\phantom{*}&space;&&space;*\phantom{*}&space;&&space;*\\&space;0&space;&&space;*\phantom{*}&space;&&space;\phantom{*}*&space;&&space;\phantom{*}*\phantom{*}&space;&&space;*\phantom{*}&space;&&space;*\\&space;1&space;&&space;2\phantom{*}&space;&&space;\phantom{*}*&space;&&space;\phantom{*}*\phantom{*}&space;&&space;*\phantom{*}&space;&&space;*\\&space;3&space;&&space;4\phantom{*}&space;&&space;\phantom{*}5&space;&&space;\phantom{*}*\phantom{*}&space;&&space;*\phantom{*}&space;&&space;*\\&space;\vdots&space;&&space;&&space;&&space;\ddots&space;&&space;\ddots&space;&\vdots\vspace*{2mm}\\&space;\tfrac{n(n-3)}{2}&space;&plus;2&space;&&space;&\phantom{*}\dots&space;&&space;&&space;\tfrac{n(n-1)}{2}\phantom{*}&space;&&space;*&space;\end{pmatrix}&space;\,." title="A_n = \begin{pmatrix} * & *\phantom{*} & \phantom{*}* & \phantom{*}*\phantom{*} & *\phantom{*} & *\\ 0 & *\phantom{*} & \phantom{*}* & \phantom{*}*\phantom{*} & *\phantom{*} & *\\ 1 & 2\phantom{*} & \phantom{*}* & \phantom{*}*\phantom{*} & *\phantom{*} & *\\ 3 & 4\phantom{*} & \phantom{*}5 & \phantom{*}*\phantom{*} & *\phantom{*} & *\\ \vdots & & & \ddots & \ddots &\vdots\vspace*{2mm}\\ \tfrac{n(n-3)}{2} +2 & &\phantom{*}\dots & & \tfrac{n(n-1)}{2}\phantom{*} & * \end{pmatrix} \,." /></a>
</p>
<br>

The mapping can be done without explicit representation of `A_n` by solving a vectorized quadratic equation. Per sample of the random graphs, this results in `m` edge tuples of form `(i, j)` and represented as a `np.ndarray`.
