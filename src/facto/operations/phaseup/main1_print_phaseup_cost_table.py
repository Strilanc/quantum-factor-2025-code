from __future__ import annotations

from facto.operations.phaseup._phaseup_coster import PhaseupCoster


def main():
    print(r"""    \begin{tabular}{|c|c|c|c|c|c|}""")
    print(
        r"""        \hline n & Workspace   & Wandering    & Multi-Target  & Reaction  & Execution Time"""
    )
    print(
        r"""        \\       & Qubits      & AND Gates    & CZ Gates      & Depth     & (approximate)"""
    )
    for n in range(1, 11):
        coster = PhaseupCoster(
            addr_len=n,
            reaction_micros=10,
            surgery_micros=25,
            factory_micros=150,
            factory_count=6,
        )
        multi_czs = coster.multi_czs
        ands = coster.ands
        reactions = coster.reaction_depth
        workspace = coster.workspace
        execution_time = coster.phaseup_duration_micros
        execution_time = round(execution_time / 10) * 10
        execution_time = rf"{{{execution_time}}}us"
        if n == 6:
            execution_time = rf"\textbf{{{execution_time}}}"
            reactions = rf"\textbf{{{reactions}}}"
            multi_czs = rf"\textbf{{{multi_czs}}}"
            ands = rf"\textbf{{{ands}}}"
            workspace = rf"\textbf{{{workspace}}}"
            n = rf"\textbf{{{n}}}"
        print(
            rf"""        \\\hline {n} & {workspace} & {ands} & {multi_czs} & {reactions} & {execution_time}"""
        )
    print(
        r"""        \\\hline n & $\leq 2 \sqrt{2^n}$ & $\leq 2 \sqrt{2^n}$ & $\leq \sqrt{2^n}$ & $(n + 3) // 2$ & $\begin{aligned} \approx& \text{AND}\cdot M \\&+ (\text{MultiCZ} + n) \cdot S \\&+\text{Reaction}\cdot R \end{aligned} $"""
    )

    print(r"""        \\\hline""")
    print(r"""    \end{tabular}""")


if __name__ == "__main__":
    main()
