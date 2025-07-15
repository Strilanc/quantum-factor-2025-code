# Log files from runs with Qunundrum
This directory contains directories which in turn contain log files from runs with [Qunundrum](https://github.com/ekera/qunundrum).

More specifically, for various parameterizations, the expected number of runs required to reach a 99% success probability in the classical lattice-based post-processing is estimated as a function of the tradeoff factor $s$.
This when solving directly without enumerating a large number of vectors in the lattice.

The estimates are based on the Gaussian heuristic.
A set of validating run, in which sampled output sets will actually be solved in practice, are planned.
(This approach is in analogy with the approach to estimating the number of runs required previously taken in [[E20]](https://doi.org/10.1007/s10623-020-00783-2), [[E21]](https://doi.org/10.1515/jmc-2020-0006), [[E19p]](https://doi.org/10.48550/arXiv.1905.09084) and [[E24t]](https://kth.diva-portal.org/smash/get/diva2:1902626/FULLTEXT01.pdf).)

For each modulus length and problem type, estimates are provided for $s = 2, 3, \ldots$ up to $s = 10$, or up to the least $s > 10$ such that at least $s + 10$ runs are required.

- The log files in [<code>rsa</code>](rsa) are for the RSA IFP with Ekerå–Håstad's algorithm [[EH17]](https://doi.org/10.1007/978-3-319-59879-6_20) with the post-processing in [[E20]](https://doi.org/10.1007/s10623-020-00783-2).

   The reduction from the RSA IFP to the short DLP is as described in App. A.2 of [[E20]](https://doi.org/10.1007/s10623-020-00783-2) and in Sect. 5.7.3 of [[E24t]](https://kth.diva-portal.org/smash/get/diva2:1902626/FULLTEXT01.pdf).
   The bit length of the logarithm is $m=l/2-1$ for $l$ the bit length of the RSA modulus.

- The log files in [<code>ff-dh-short</code>](ff-dh-short) are for the short DLP in safe-prime groups with Ekerå–Håstad's algorithm [[EH17]](https://doi.org/10.1007/978-3-319-59879-6_20) with the post-processing in [[E20]](https://doi.org/10.1007/s10623-020-00783-2).

   The strength level in bits $z$ is computed according to the NIST model, see Sect. 7.2.1.1 and Tab. 7.3 on p. 130 of [[E24t]](https://kth.diva-portal.org/smash/get/diva2:1902626/FULLTEXT01.pdf) for further details.
   The bit length of the logarithm is $m = 2z$, and $\ell = \lceil m / s \rceil$.

- The log files in [<code>ff-dh-schnorr</code>](ff-dh-schnorr) are for the DLP in Schnorr groups of known order when solving with Shor's algorithm [[Shor94]](https://doi.org/10.1109/SFCS.1994.365700) [[Shor97]](https://doi.org/10.1137/S0097539795293172) as modified by Ekerå in [[E19p]](https://doi.org/10.48550/arXiv.1905.09084) and with the post-processing in [[E19p]](https://doi.org/10.48550/arXiv.1905.09084).

   The strength level in bits $z$ is computed according to the NIST model, see Sect. 7.2.1.1 and Tab. 7.3 on p. 130 of [[E24t]](https://kth.diva-portal.org/smash/get/diva2:1902626/FULLTEXT01.pdf) for further details.
   The bit length of the logarithm is $m = 2z$, and $\ell = \lceil m / s \rceil$.

   The parameter $\varsigma$ is selected as in Tab. 5.6 on p. 82 of [[E24t]](https://kth.diva-portal.org/smash/get/diva2:1902626/FULLTEXT01.pdf) with $B_q = 0.991$.

   Note that the estimates in these log files are based on heuristics from [[E19p]](https://doi.org/10.48550/arXiv.1905.09084) that may not be good for large $s$.