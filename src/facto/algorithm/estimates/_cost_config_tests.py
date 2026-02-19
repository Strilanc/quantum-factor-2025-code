from __future__ import annotations

import collections
import sys
import random
from typing import Mapping

from facto.algorithm.estimates._cost_config import min_pareto_frontier
from facto.algorithm.prep import table_str, ExecutionConfig
from facto.algorithm.prep._execution_config import black_box_costs, simplified_costs
from facto.algorithm._detailed_example_code import loop1, loop2, loop3, loop4, unloop3, unloop2
from scatter_script import QPU, CostKey, Lookup


def assert_costs_comparable(
    *, actual_costs: Mapping[CostKey, int], estimated_costs: Mapping[CostKey, int]
):
    actual_costs = dict(actual_costs)
    actual_costs.pop(CostKey("CX"), None)
    actual_costs.pop(CostKey("CZ"), None)
    actual_costs.pop(CostKey("qubits"), None)

    discrepancies = []
    for k in actual_costs.keys() | estimated_costs.keys():
        a = actual_costs[k]
        e = estimated_costs[k]
        if not (0.9 * e <= a <= 1.1 * e):
            discrepancies.append((k, a, e))
    if not discrepancies:
        return

    row = 0
    table = {}
    table[-2j + 0] = "key"
    table[-2j + 1] = "actual"
    table[-2j + 2] = "expected"
    table[-1j + 0] = "---"
    table[-1j + 1] = "---"
    table[-1j + 2] = "---"
    for k in sorted(actual_costs.keys() | estimated_costs.keys()):
        a = actual_costs[k]
        e = estimated_costs[k]
        table[row * 1j + 0] = k
        table[row * 1j + 1] = a
        table[row * 1j + 2] = e
        if not (0.9 * e <= a <= 1.1 * e):
            table[row * 1j + 3] = "<<<<<<<<<<<<<< DISCREPANCY <<<<<<<<<<<<<"
        row += 1

    table[row * 1j + 0] = "---"
    table[row * 1j + 1] = "---"
    table[row * 1j + 2] = "---"
    row += 1

    actual_simplified = simplified_costs(actual_costs)
    estimated_simplified = simplified_costs(estimated_costs)
    for k in sorted(actual_simplified.keys() | estimated_simplified.keys()):
        a = actual_simplified[k]
        e = estimated_simplified[k]
        table[row * 1j + 0] = k
        table[row * 1j + 1] = a
        table[row * 1j + 2] = e
        if not (0.9 * e <= a <= 1.1 * e):
            table[row * 1j + 3] = "<<<<<<<<<<<<<< DISCREPANCY <<<<<<<<<<<<<"
        row += 1

    table[row * 1j + 0] = "---"
    table[row * 1j + 1] = "---"
    table[row * 1j + 2] = "---"
    row += 1

    actual_black_box = black_box_costs(actual_costs, reaction_us=10, clifford_us=25)
    estimated_black_box = black_box_costs(estimated_costs, reaction_us=10, clifford_us=25)
    for k in sorted(actual_black_box.keys() | estimated_black_box.keys()):
        a = actual_black_box[k]
        e = estimated_black_box[k]
        table[row * 1j + 0] = k
        table[row * 1j + 1] = a
        table[row * 1j + 2] = e
        if not (0.9 * e <= a <= 1.1 * e):
            table[row * 1j + 3] = "<<<<<<<<<<<<<< DISCREPANCY <<<<<<<<<<<<<"
        row += 1

    print("\n" + table_str(table), file=sys.stderr)
    assert not discrepancies, "Cost discrepancies were present:\n\n" + table_str(table)


def test_costs_loop1():
    len_acc = random.randrange(5, 15)
    conf = ExecutionConfig.vacuous_config(
        modulus_bitlength=len_acc,
        period_bitlength=random.randrange(5, 10),
        len_acc=len_acc,
        w1=random.randrange(2, 4),
        num_input_qubits=10,
        num_periods=random.randrange(32, 64),
    )
    qpu = QPU(num_branches=1)
    shots = 50
    for _ in range(shots):
        loop1(
            Q_dlog=qpu.alloc_quint(length=conf.len_dlog_accumulator),
            conf=conf,
            Q_exponent=qpu.alloc_quint(length=conf.num_input_qubits),
            i=0,
            vent=Lookup(conf.table1).venting_into_new_table().vent,
        )
    actual_costs = collections.Counter()
    for k, v in qpu.cost_counters.items():
        actual_costs[k] = v / shots

    estimated_costs = conf.estimated_subroutine_costs_per_call["loop1"]

    assert_costs_comparable(actual_costs=actual_costs, estimated_costs=estimated_costs)


def test_costs_loop2():
    len_acc = random.randrange(5, 15)
    conf = ExecutionConfig.vacuous_config(
        modulus_bitlength=len_acc,
        period_bitlength=random.randrange(5, 10),
        len_acc=len_acc,
        num_input_qubits=10,
        num_periods=random.randrange(32, 64),
    )
    qpu = QPU(num_branches=1)
    shots = 50
    for _ in range(shots):
        loop2(
            Q_target=qpu.alloc_quint(length=conf.len_dlog_accumulator),
            modulus=int(conf.periods[0]),
            compressed_len=conf.rns_primes_bit_length,
        )
    actual_costs = collections.Counter()
    for k, v in qpu.cost_counters.items():
        actual_costs[k] = v / shots

    estimated_costs = conf.estimated_subroutine_costs_per_call["loop2"]

    assert_costs_comparable(actual_costs=actual_costs, estimated_costs=estimated_costs)


def test_costs_loop3():
    len_acc = random.randrange(5, 15)
    conf = ExecutionConfig.vacuous_config(
        modulus_bitlength=len_acc,
        period_bitlength=random.randrange(5, 10),
        len_acc=len_acc,
        w3a=random.randrange(2, 4),
        w3b=random.randrange(2, 4),
        num_input_qubits=10,
        num_periods=1,
    )
    qpu = QPU(num_branches=1)
    shots = 50
    for _ in range(shots):
        loop3(qpu=qpu, Q_dlog=qpu.alloc_quint(length=conf.len_dlog_accumulator), conf=conf, i=0)
    actual_costs = collections.Counter()
    for k in qpu.cost_counters.keys():
        v = qpu.cost_counters[k] / shots
        actual_costs[k] = v

    estimated_costs = conf.estimated_subroutine_costs_per_call["loop3"]

    assert_costs_comparable(actual_costs=actual_costs, estimated_costs=estimated_costs)


def test_costs_loop4():
    len_acc = random.randrange(10, 30)
    conf = ExecutionConfig.vacuous_config(
        modulus_bitlength=len_acc,
        period_bitlength=random.randrange(15, 25),
        len_acc=len_acc,
        w4=random.randrange(1, 5),
        num_input_qubits=10,
        num_periods=1,
    )
    qpu = QPU(num_branches=1)
    shots = 1000
    for _ in range(shots):
        loop4(
            qpu=qpu,
            Q_residue=qpu.alloc_quint(length=conf.rns_primes_bit_length),
            Q_acc=qpu.alloc_quint(length=conf.len_accumulator + 1),
            conf=conf,
            i=0,
        )
    actual_costs = collections.Counter()
    for k in qpu.cost_counters.keys():
        v = qpu.cost_counters[k] / shots
        actual_costs[k] = v

    estimated_costs = conf.estimated_subroutine_costs_per_call["loop4"]

    assert_costs_comparable(actual_costs=actual_costs, estimated_costs=estimated_costs)


def test_costs_unloop3():
    len_acc = random.randrange(5, 15)
    conf = ExecutionConfig.vacuous_config(
        modulus_bitlength=len_acc,
        period_bitlength=random.randrange(5, 10),
        len_acc=len_acc,
        w3a=random.randrange(2, 4),
        w3b=random.randrange(2, 4),
        num_input_qubits=10,
        num_periods=1,
    )
    qpu = QPU(num_branches=1)
    shots = 50
    total = collections.Counter()
    for _ in range(shots):
        Q_result = loop3(
            qpu=qpu, Q_dlog=qpu.alloc_quint(length=conf.len_dlog_accumulator), conf=conf, i=0
        )
        qpu.cost_counters.clear()
        unloop3(
            qpu=qpu,
            Q_dlog=qpu.alloc_quint(length=conf.len_dlog_accumulator),
            conf=conf,
            i=0,
            Q_unresult=Q_result,
        )
        total += qpu.cost_counters
    actual_costs = collections.Counter()
    for k, v in total.items():
        actual_costs[k] = v / shots

    estimated_costs = conf.estimated_subroutine_costs_per_call["unloop3"]

    assert_costs_comparable(actual_costs=actual_costs, estimated_costs=estimated_costs)


def test_costs_unloop2():
    len_acc = random.randrange(5, 15)
    conf = ExecutionConfig.vacuous_config(
        modulus_bitlength=len_acc,
        period_bitlength=random.randrange(5, 10),
        len_acc=len_acc,
        num_input_qubits=10,
        num_periods=random.randrange(32, 64),
    )
    qpu = QPU(num_branches=1)
    shots = 50
    for _ in range(shots):
        unloop2(
            Q_target=qpu.alloc_quint(length=conf.len_dlog_accumulator),
            modulus=int(conf.periods[0]),
            compressed_len=conf.rns_primes_bit_length,
        )
    actual_costs = collections.Counter()
    for k, v in qpu.cost_counters.items():
        actual_costs[k] = v / shots

    estimated_costs = conf.estimated_subroutine_costs_per_call["unloop2"]

    assert_costs_comparable(actual_costs=actual_costs, estimated_costs=estimated_costs)


def test_pareto_frontier():
    assert min_pareto_frontier([(0, 0), (1, 0), (0, 1), (1, 1)]) == [(0, 0)]
    assert min_pareto_frontier([(3, 0), (2, 1), (1, 2), (0, 3)]) == [(3, 0), (2, 1), (1, 2), (0, 3)]
    assert min_pareto_frontier([(3, 0), (3, 1), (3, 2), (0, 3)]) == [(3, 0), (0, 3)]
