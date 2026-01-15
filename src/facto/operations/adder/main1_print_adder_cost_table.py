from __future__ import annotations

from facto.operations.adder._add_coster import AddCoster


def main():
    print(r"""    \begin{tabular}{|c|c|c|c|c|}""")
    print(r"""        \hline n & Compute    & Uncompute  & CCZ States & Execution Time""")
    print(r"""        \\       & Layers (A) & Layers (B) & (C)        & (approximate)""")
    for n in range(5, 51, 5):
        coster = AddCoster(
            n=n,
            reaction_micros=10,
            surgery_micros=25,
            factory_micros=150,
            factory_count=6,
        )
        compute_layers = coster.adder_compute_layers
        uncompute_layers = coster.adder_uncompute_layers
        ccz_states = coster.adder_ccz_states
        execution_time = coster.adder_duration_micros

        execution_time = round(execution_time / 10) * 10
        execution_time = rf"{{{execution_time}}}us"
        if n == 35:
            compute_layers = rf"\textbf{{{compute_layers}}}"
            uncompute_layers = rf"\textbf{{{uncompute_layers}}}"
            ccz_states = rf"\textbf{{{ccz_states}}}"
            execution_time = rf"\textbf{{{execution_time}}}"
            n = rf"\textbf{{{n}}}"
        print(
            rf"""        \\\hline {n} """
            rf"""& {compute_layers} """
            rf"""& {uncompute_layers} """
            rf"""& {ccz_states} """
            rf"""& {execution_time}"""
        )
    print(
        r"""        \\\hline n """
        r"""& $6n / F$ """
        r"""& $2n / F$ """
        r"""& $n - 1$ """
        r"& $\begin{aligned} \approx &\text{max}(B \cdot S, C \cdot R) "
        r"\\&+ \text{max}(A \cdot S, C \cdot R, C \cdot M)"
        r"\end{aligned} $"
    )

    print(r"""        \\\hline""")
    print(r"""    \end{tabular}""")


if __name__ == "__main__":
    main()
