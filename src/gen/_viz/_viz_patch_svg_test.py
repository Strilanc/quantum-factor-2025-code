import gen


def test_patch_svg_runs():
    patch = gen.Patch(
        tiles=[
            gen.Tile(data_qubits=(None, 1j, None, 2j), measure_qubit=(-0.5 + 1.5j), bases="Z"),
            gen.Tile(data_qubits=(None, 0j, None, (1 + 0j)), measure_qubit=(0.5 - 0.5j), bases="X"),
            gen.Tile(
                data_qubits=(0j, (1 + 0j), 1j, (1 + 1j)), measure_qubit=(0.5 + 0.5j), bases="Z"
            ),
            gen.Tile(
                data_qubits=(1j, 2j, (1 + 1j), (1 + 2j)), measure_qubit=(0.5 + 1.5j), bases="X"
            ),
            gen.Tile(
                data_qubits=((1 + 0j), (1 + 1j), (2 + 0j), (2 + 1j)),
                measure_qubit=(1.5 + 0.5j),
                bases="X",
            ),
            gen.Tile(
                data_qubits=((1 + 1j), (2 + 1j), (1 + 2j), (2 + 2j)),
                measure_qubit=(1.5 + 1.5j),
                bases="Z",
            ),
            gen.Tile(
                data_qubits=((1 + 2j), None, (2 + 2j), None), measure_qubit=(1.5 + 2.5j), bases="X"
            ),
            gen.Tile(
                data_qubits=((2 + 0j), None, (2 + 1j), None), measure_qubit=(2.5 + 0.5j), bases="Z"
            ),
        ]
    )
    svg_content = gen.svg([patch])
    assert svg_content is not None
