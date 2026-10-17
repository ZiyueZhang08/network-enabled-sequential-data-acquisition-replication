# Entropy Eq.3.6 Equivalence Note: paper_faithful_entropy_eq36_equivalence_v1

## Result

The implemented Schur entropy scorer is numerically equivalent to the
explicit observed-kernel Schur complement on a non-orthogonal toy
factorization. The maximum absolute difference in this audit is
`2.220e-16`.

## What Was Verified

The direct kernel form is:

```text
k_xx - k_xO @ inv(K_OO + lambda I) @ k_Ox
```

The production scorer uses the equivalent feature-space form:

```text
lambda * phi_x.T @ inv(Phi_O.T @ Phi_O + lambda I) @ phi_x
```

Here `phi_x = kron(q_user[u], q_item[i])`, where `q_user` and
`q_item` are orthonormal bases derived from the DLS user factors and
item factors. The equivalence follows from the Woodbury identity and
is now locked by `tests/test_core.py::test_schur_entropy_matches_explicit_kernel_schur_complement`.

## Boundary

This closes the formula-implementation risk for the Schur conditional
variance used by the clean-room DU-Net proxy. It does not recover
unpublished author masks, the exact Figure 4 metric normalization, or
the paper's exact Linear UCB implementation.

## Audit Values

| user_id | item_id | rating | lambda_noise | feature_space_value | explicit_kernel_schur_value | absolute_difference |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 5 | 0.75 | 0.471891024299 | 0.471891024299 | 1.11022302463e-16 |
| 0 | 2 | 4 | 0.75 | 0.181161167765 | 0.181161167765 | 2.77555756156e-17 |
| 1 | 1 | 3 | 0.75 | 0.878059243554 | 0.878059243554 | 2.22044604925e-16 |
| 2 | 2 | 2 | 0.75 | 0.0859851519925 | 0.0859851519925 | 1.38777878078e-17 |
