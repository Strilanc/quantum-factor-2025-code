# Sets of algorithm parameterizations
This directory contains sets of algorithm parameterizations on CSV format for several different cryptographically relevant problems.
More specifically, the parameterizations are for:

- The RSA IFP with
   - Ekerå–Håstad's algorithm [[EH17]](https://doi.org/10.1007/978-3-319-59879-6_20) with the post-processing in [[E20]](https://doi.org/10.1007/s10623-020-00783-2) for $s \ge 2$, and 
   - Ekerå–Håstad's algorithm [[EH17]](https://doi.org/10.1007/978-3-319-59879-6_20) with the post-processing in [[E23p]](https://doi.org/10.48550/arXiv.2309.01754) for $s \approx 1$.

- The short DLP in safe-prime groups with
   - Ekerå–Håstad's algorithm [[EH17]](https://doi.org/10.1007/978-3-319-59879-6_20) with the post-processing in [[E20]](https://doi.org/10.1007/s10623-020-00783-2) for $s \ge 2$, and 
   - Ekerå–Håstad's algorithm [[EH17]](https://doi.org/10.1007/978-3-319-59879-6_20) with the post-processing in [[E23p]](https://doi.org/10.48550/arXiv.2309.01754) for $s \approx 1$.

- The DLP in Schnorr groups of known order with
   - Shor's algorithm [[Shor94]](https://doi.org/10.1109/SFCS.1994.365700) [[Shor97]](https://doi.org/10.1137/S0097539795293172) as modified by Ekerå in [[E19p]](https://doi.org/10.48550/arXiv.1905.09084) and with the post-processing in [[E19p]](https://doi.org/10.48550/arXiv.1905.09084).

   Note that these parameterizations are based on heuristics from [[E19p]](https://doi.org/10.48550/arXiv.1905.09084) that may not be good for large $s$.

The algorithm parameterizations can be passed to the cost estimation shell script in the parent directory to find and tabulate optimal parameterizations and to generate Pareto plots. Above, $s$ refers to the so-called tradeoff factor in the aforementioned works.

The algorithm parameterizations were in part generated by the [Qunundrum](https://github.com/ekera/qunundrum) suite of MPI programs (see the [<code>logs</code>](logs)) and then converted to CSV files (see the [<code>create-csv-files.py</code>](create-csv-files.py) Python script).
More specifically, this approach was taken for all parameterizations with $s \ge 2$.

For the parameterizations with $s \approx 1$ data was instead fetched from [[E19p]](https://doi.org/10.48550/arXiv.1905.09084) and [[E23p]](https://doi.org/10.48550/arXiv.2309.01754) directly, as well as from [[E24t]](https://kth.diva-portal.org/smash/get/diva2:1902626/FULLTEXT01.pdf).

## About and acknowledgments
The contribution in this directory was made by Martin Ekerå, in part at [KTH, the Royal Institute of Technology](https://www.kth.se/en), in Stockholm, [Sweden](https://www.sweden.se).
Funding and support for this work was provided by the Swedish NCSA that is a part of the [Swedish Armed Forces](https://www.mil.se).