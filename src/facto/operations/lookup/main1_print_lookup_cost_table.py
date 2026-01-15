from __future__ import annotations

from facto.operations.lookup._lookup_coster import LookupCoster


def main():
    print(r"""    \begin{tabular}{|c|c|c|c|c|c|c|}""")
    print(
        r"""        \hline n & Workspace &Multi-Target& Multi-Target   & Power-Product  & Reaction  & Execution Time"""
    )
    print(
        r"""        \\       & Qubits    &CX Gates (A)& CCX Gates (B)  & AND Gates (C)  & Depth (D) & (approximate)"""
    )
    for n in range(1, 11):
        coster = LookupCoster(
            addr_len=n,
            reaction_micros=10,
            surgery_micros=25,
            factory_micros=150,
            factory_count=6,
        )
        power_product_and_gates = coster.and_gates_for_power_product
        multi_target_ccx_gates = coster.multi_target_ccx_gates
        multi_target_cx_gates = coster.multi_target_cx_gates
        reactions = coster.reaction_depth
        workspace = coster.workspace
        execution_time = coster.lookup_duration_micros

        execution_time = round(execution_time / 10) * 10
        execution_time = rf"{{{execution_time}}}us"
        if n == 6:
            execution_time = rf"\textbf{{{execution_time}}}"
            reactions = rf"\textbf{{{reactions}}}"
            multi_target_ccx_gates = rf"\textbf{{{multi_target_ccx_gates}}}"
            multi_target_cx_gates = rf"\textbf{{{multi_target_cx_gates}}}"
            power_product_and_gates = rf"\textbf{{{power_product_and_gates}}}"
            workspace = rf"\textbf{{{workspace}}}"
            n = rf"\textbf{{{n}}}"
        print(
            rf"""        \\\hline {n} """
            rf"""& {workspace} """
            rf"""& {multi_target_cx_gates} """
            rf"""& {multi_target_ccx_gates} """
            rf"""& {power_product_and_gates} """
            rf"""& {reactions} """
            rf"""& {execution_time}"""
        )
    print(
        r"""        \\\hline n """
        r"""& $\leq 2 \sqrt{2^n}$ """
        r"""& $\leq 2 \sqrt{2^n}$ """
        r"""& $\leq 2^n - 2n$ """
        r"""& $\leq 2 \sqrt{2^n}$ """
        r"""& $\leq 2^n - 2n$ """
        r"""& $\begin{aligned} \approx"""
        r"& A \cdot S "
        r"\\&+ B \cdot \text{max}(M, R, S) "
        r"\\& + C \cdot M "
        r"\\& + R + nS"
        r"\end{aligned} $"
    )

    print(r"""        \\\hline""")
    print(r"""    \end{tabular}""")


if __name__ == "__main__":
    main()
