multi_mnist = dict(
    dataset='multi_mnist',
    n_mo_sol = 16,
    lamda=5.,
    ref_point = (2, 2),
    n_mo_obj = 2,
    partition = True
)

multi_fashion = dict(
    dataset='multi_fashion',
    n_mo_sol = 16,
    lamda=4.,
    ref_point = (2, 2),
    n_mo_obj = 2,
    partition = True
)

multi_fashion_mnist = dict(
    dataset='multi_fashion_mnist',
    n_mo_sol = 16,
    lamda=4.,
    ref_point = (2, 2),
    n_mo_obj = 2,
    partition=False
)