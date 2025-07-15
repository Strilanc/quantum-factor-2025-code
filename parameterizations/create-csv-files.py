import csv
import json

from math import ceil

ff_dh_short_dlp_params = [
    # modulus length, strength level, [s, runs] combinations
    # (for >= 99% success probability)
    [ 512,  56, # => 112-bit logarithm
        [[ 2,  3],
         [ 3,  4],
         [ 4,  6],
         [ 5,  8],
         [ 6, 11],
         [ 7, 16],
         [ 8, 22],
         [ 9, 28],
         [10, 37]]],
    [ 768,  72, # => 144-bit logarithm
        [[ 2,  3],
         [ 3,  4],
         [ 4,  6],
         [ 5,  7],
         [ 6, 10],
         [ 7, 12],
         [ 8, 16],
         [ 9, 20],
         [10, 24]]],
    [1024,  80, # => 160-bit logarithm
        [[ 2,  3],
         [ 3,  4],
         [ 4,  6],
         [ 5,  7],
         [ 6,  9],
         [ 7, 12],
         [ 8, 14],
         [ 9, 18],
         [10, 23]]],
    [1536,  96, # => 192-bit logarithm
        [[ 2,  3],
         [ 3,  4],
         [ 4,  5],
         [ 5,  7],
         [ 6,  9],
         [ 7, 10],
         [ 8, 13],
         [ 9, 15],
         [10, 17],
         [11, 21]]],
    [2048, 112, # => 224-bit logarithm
        [[ 2,  3],
         [ 3,  4],
         [ 4,  5],
         [ 5,  7],
         [ 6,  8],
         [ 7, 10],
         [ 8, 12],
         [ 9, 14],
         [10, 16],
         [11, 19],
         [12, 22]]],
    [3072, 128, # => 256-bit logarithm
        [[ 2,  3],
         [ 3,  4],
         [ 4,  5],
         [ 5,  6],
         [ 6,  8],
         [ 7,  9],
         [ 8, 11],
         [ 9, 13],
         [10, 15],
         [11, 17],
         [12, 20],
         [13, 23]]],
    [4096, 152, # => 304-bit logarithm
        [[2,   3],
         [3,   4],
         [4,   5],
         [5,   6],
         [6,   8],
         [7,   9],
         [8,  11],
         [9,  12],
         [10, 14],
         [11, 16],
         [12, 18],
         [13, 20],
         [14, 24]]],
    [6144, 176, # => 352-bit logarithm
        [[ 2,  3],
         [ 3,  4],
         [ 4,  5],
         [ 5,  6],
         [ 6,  7],
         [ 7,  9],
         [ 8, 10],
         [ 9, 12],
         [10, 13],
         [11, 16],
         [12, 17],
         [13, 19],
         [14, 21],
         [15, 24],
         [16, 27]]],
    [8192, 200, # => 400-bit logarithm
        [[ 2,  3],
         [ 3,  4],
         [ 4,  5],
         [ 5,  6],
         [ 6,  7],
         [ 7,  9],
         [ 8, 10],
         [ 9, 11],
         [10, 13],
         [11, 15],
         [12, 16],
         [13, 18],
         [14, 20], 
         [15, 22],
         [16, 25],
         [17, 27]]],
]

ff_dh_short_dlp_single_options = [
    # delta, tau, t, success probability, complexity
    [50, 10, 29,       0.999, 33.6],
    [30, 14, 17,  1 - 10**-4, 25.6],
    [20, 20, 14,  1 - 10**-6, 23.6],
    [ 0, 32,  2, 1 - 10**-10, 22.1]
]

ff_dh_schnorr_dlp_params = [
    # modulus length, strength level, [s, runs, sigma] combinations
    # (for >= 99% success probability)
    [ 512,  56, # => 112-bit logarithm
        [[ 2,  3,  8],
         [ 3,  4,  8],
         [ 4,  6,  9],
         [ 5,  8,  9],
         [ 6, 11,  9],
         [ 7, 14, 10],
         [ 8, 20, 10],
         [ 9, 23, 11],
         [10, 32, 11]]],
    [ 768,  72, # => 144-bit logarithm
        [[ 2,  3,  8],
         [ 3,  4,  8],
         [ 4,  6,  9],
         [ 5,  7,  9],
         [ 6,  9,  9],
         [ 7, 11, 10], # or [ 7, 12,  9]
         [ 8, 15, 10],
         [ 9, 19, 10],
         [10, 21, 11]]], # or [10, 23, 10]
    [1024,  80, # => 160-bit logarithm
        [[ 2,  3,  8],
         [ 3,  4,  8],
         [ 4,  5,  8],
         [ 5,  7,  9],
         [ 6,  9,  9],
         [ 7, 11, 10],
         [ 8, 14, 10],
         [ 9, 17, 10],
         [10, 22, 10]]],
    [1536,  96, # => 192-bit logarithm
        [[ 2,  3,  8],
         [ 3,  4,  8],
         [ 4,  5,  8],
         [ 5,  7,  9],
         [ 6,  8,  9],
         [ 7, 10,  9],
         [ 8, 12, 10],
         [ 9, 14, 10],
         [10, 17, 10],
         [11, 20, 10],
         [12, 25, 11]]],
    [2048, 112, # => 224-bit logarithm
        [[ 2,  3,  8],
         [ 3,  4,  8],
         [ 4,  5,  8],
         [ 5,  6,  9],
         [ 6,  8,  9],
         [ 7, 10,  9],
         [ 8, 12, 10],
         [ 9, 14, 10],
         [10, 15, 10],
         [11, 18, 10],
         [12, 22, 10]]],
    [3072, 128, # => 256-bit logarithm
        [[ 2,  3,  8],
         [ 3,  4,  8],
         [ 4,  5,  8],
         [ 5,  6,  9],
         [ 6,  8,  9],
         [ 7,  9,  9],
         [ 8, 11,  9],
         [ 9, 13, 10],
         [10, 15, 10],
         [11, 17, 10], 
         [12, 19, 10],
         [13, 22, 11], # or [13, 23, 10]
         [14, 24, 11]]],
    [4096, 152, # => 304-bit logarithm
        [[ 2,  3,  8],
         [ 3,  4,  8],
         [ 4,  5,  8],
         [ 5,  6,  9],
         [ 6,  8,  9],
         [ 7,  9,  9],
         [ 8, 11,  9],
         [ 9, 12, 10],
         [10, 14, 10],
         [11, 16, 10],
         [12, 18, 10],
         [13, 20, 10],
         [14, 22, 11], # or [14, 23, 10]
         [15, 24, 11],
         [16, 29, 11]]],
    [6144, 176, # => 352-bit logarithm
        [[ 2,  3,  8],
         [ 3,  4,  8],
         [ 4,  5,  8],
         [ 5,  6,  9],
         [ 6,  7,  9],
         [ 7,  9,  9],
         [ 8, 10,  9],
         [ 9, 11, 10], # or [9, 12, 9],
         [10, 13, 10],
         [11, 15, 10],
         [12, 17, 10],
         [13, 18, 10],
         [14, 21, 10],
         [15, 23, 11],
         [16, 26, 11]]],
    [8192, 200, # => 400-bit logarithm
        [[ 2,  3,  8],
         [ 3,  4,  8],
         [ 4,  5,  8],
         [ 5,  6,  9],
         [ 6,  7,  9],
         [ 7,  8,  9],
         [ 8, 10,  9],
         [ 9, 11, 10],
         [10, 13, 10],
         [11, 14, 10],
         [12, 16, 10],
         [13, 18, 10],
         [14, 20, 10],
         [15, 22, 10],
         [16, 24, 11],
         [17, 26, 11],
         [18, 28, 11]]],
]

rsa_params = [
    # modulus length, [s, runs] combinations for >= 99% success probability
    [ 512, # => 255-bit logarithm
        [[ 2,  3],
         [ 3,  4],
         [ 4,  5],
         [ 5,  6],
         [ 6,  8],
         [ 7,  9],
         [ 8, 11],
         [ 9, 13],
         [10, 15],
         [11, 17],
         [12, 20],
         [13, 23]]],
    [ 768, # => 383-bit logarithm
        [[ 2,  3],
         [ 3,  4],
         [ 4,  5],
         [ 5,  6],
         [ 6,  7],
         [ 7,  9],
         [ 8, 10],
         [ 9, 12],
         [10, 13],
         [11, 15],
         [12, 17],
         [13, 18],
         [14, 20],
         [15, 23],
         [16, 26]]],
    [1024, # => 511-bit logarithm
        [[ 2,  3],
         [ 3,  4],
         [ 4,  5],
         [ 5,  6],
         [ 6,  7],
         [ 7,  8],
         [ 8, 10],
         [ 9, 11],
         [10, 12],
         [11, 14],
         [12, 15],
         [13, 17],
         [14, 19],
         [15, 20],
         [16, 23],
         [17, 24],
         [18, 26],
         [19, 29]]],
    [1536, # => 767-bit logarithm
        [[ 2,  3],
         [ 3,  4],
         [ 4,  5],
         [ 5,  6],
         [ 6,  7],
         [ 7,  8],
         [ 8,  9],
         [ 9, 10],
         [10, 12],
         [11, 13],
         [12, 14],
         [13, 16],
         [14, 17],
         [15, 18],
         [16, 20],
         [17, 21],
         [18, 23],
         [19, 24],
         [20, 26],
         [21, 28],
         [22, 30],
         [23, 31],
         [24, 34]]],
    [2048, # => 1023-bit logarithm
        [[ 2,  3],
         [ 3,  4],
         [ 4,  5],
         [ 5,  6],
         [ 6,  7],
         [ 7,  8],
         [ 8,  9],
         [ 9, 10], 
         [10, 11],
         [11, 13],
         [12, 14],
         [13, 15],
         [14, 16],
         [15, 17],
         [16, 19],
         [17, 20],
         [18, 22],
         [19, 23],
         [20, 24],
         [21, 26],
         [22, 27],
         [23, 29],
         [24, 31],
         [25, 32],
         [26, 34],
         [27, 36],
         [28, 37],
         [29, 39]]],
    [3072, # => 1535-bit logarithm
        [[ 2,  3],
         [ 3,  4],
         [ 4,  5],
         [ 5,  6],
         [ 6,  7],
         [ 7,  8],
         [ 8,  9],
         [ 9, 10],
         [10, 11],
         [11, 12],
         [12, 13],
         [13, 14],
         [14, 16],
         [15, 17],
         [16, 18],
         [17, 19],
         [18, 20],
         [19, 22],
         [20, 23],
         [21, 24],
         [22, 26],
         [23, 27],
         [24, 28],
         [25, 29],
         [26, 31],
         [27, 32],
         [28, 34],
         [29, 35],
         [30, 36],
         [31, 38],
         [32, 40],
         [33, 41],
         [34, 42],
         [35, 45]]],
    [4096, # => 2047-bit logarithm
        [[ 2,  3],
         [ 3,  4],
         [ 4,  5],
         [ 5,  6],
         [ 6,  7],
         [ 7,  8],
         [ 8,  9],
         [ 9, 10],
         [10, 11],
         [11, 12],
         [12, 13],
         [13, 14],
         [14, 15],
         [15, 16],
         [16, 18],
         [17, 19],
         [18, 20],
         [19, 21],
         [20, 22],
         [21, 23],
         [22, 25],
         [23, 26],
         [24, 27],
         [25, 29],
         [26, 30],
         [27, 31],
         [28, 32],
         [29, 34],
         [30, 35],
         [31, 36],
         [32, 38],
         [33, 38],
         [34, 40],
         [35, 42],
         [36, 43],
         [37, 44],
         [38, 46],
         [39, 47],
         [40, 48],
         [41, 51]]],
    [6144, # => 3071-bit logarithm
        [[ 2,  3],
         [ 3,  4],
         [ 4,  5],
         [ 5,  6],
         [ 6,  7],
         [ 7,  8],
         [ 8,  9],
         [ 9, 10],
         [10, 11],
         [11, 12],
         [12, 13],
         [13, 14],
         [14, 15],
         [15, 16],
         [16, 17],
         [17, 18],
         [18, 19],
         [19, 21],
         [20, 22],
         [21, 23],
         [22, 24],
         [23, 25],
         [24, 26],
         [25, 27],
         [26, 28],
         [27, 30],
         [28, 31],
         [29, 32],
         [30, 33],
         [31, 34],
         [32, 36],
         [33, 37],
         [34, 38],
         [35, 39],
         [36, 40],
         [37, 42],
         [38, 43],
         [39, 44],
         [40, 46],
         [41, 47],
         [42, 48],
         [43, 49],
         [44, 51],
         [45, 52],
         [46, 54],
         [47, 55],
         [48, 57],
         [49, 58],
         [50, 59],
         [51, 60],
         [52, 61],
         [53, 64]]],
    [8192, # => 4091-bit logarithm
        [[ 2,  3],
         [ 3,  4],
         [ 4,  5],
         [ 5,  6],
         [ 6,  7],
         [ 7,  8],
         [ 8,  9],
         [ 9, 10],
         [10, 11],
         [11, 12],
         [12, 13],
         [13, 14],
         [14, 15],
         [15, 16],
         [16, 17],
         [17, 18],
         [18, 19],
         [19, 20],
         [20, 21],
         [21, 22],
         [22, 24],
         [23, 25],
         [24, 26],
         [25, 27],
         [26, 28],
         [27, 29],
         [28, 30],
         [29, 31],
         [30, 32],
         [31, 34],
         [32, 35],
         [33, 36],
         [34, 37],
         [35, 38],
         [36, 40],
         [37, 41],
         [38, 42],
         [39, 43],
         [40, 44],
         [41, 46],
         [42, 47],
         [43, 48],
         [44, 49],
         [45, 51],
         [46, 52],
         [47, 52],
         [48, 54],
         [49, 55],
         [50, 57],
         [51, 58],
         [52, 59],
         [53, 60],
         [54, 62],
         [55, 63],
         [56, 64],
         [57, 66],
         [58, 67],
         [59, 68],
         [60, 69],
         [61, 70],
         [62, 73]]],
]

rsa_single_options = [
    # delta, tau, t, success probability, complexity
    [50, 11, 27,      0.999, 34.1],
    [30, 16, 17, 1 - 10**-4, 26.6],
    [21, 16, 12, 1 - 10**-4, 22.1],
    [20, 11, 12,      0.999, 19.1]
]


def tabulate_ff_dh_short(csv_prefix):
  with open(csv_prefix + "-params.csv", "w+") as file:
    csv_writer = csv.DictWriter(file, fieldnames=[
      "modulus_bitlength",
      "num_input_qubits",
      "num_shots",
      "pp_success_probability",
      "details"])
    csv_writer.writeheader()

    for [modulus_length, z, options] in ff_dh_short_dlp_params:
      m = 2 * z

      # Single run.
      for [delta, pp_tau, pp_t, pp_success_probability, pp_complexity] in \
        ff_dh_short_dlp_single_options:
        l = m - delta
        num_input_qubits = m + 2 * l

        csv_writer.writerow({
          "modulus_bitlength": modulus_length,
          "num_input_qubits": num_input_qubits,
          "num_shots": 1,
          "pp_success_probability": pp_success_probability,
          "details": json.dumps({
            "z": z,
            "m": m,
            "delta": delta,
            "l": l,
            "pp_tau": pp_tau,
            "pp_t": pp_t,
            "pp_complexity": pp_complexity,
            "algorithms": ["EH17", "E23p"]
          })
        })

      # Multiple runs.
      for [s, n] in options:
        l = ceil(m / s)
        num_input_qubits = m + 2 * l

        csv_writer.writerow({
          "modulus_bitlength": modulus_length,
          "num_input_qubits": num_input_qubits,
          "num_shots": n,
          "pp_success_probability": 0.99,
          "details": json.dumps({
            "z": z,
            "m": m,
            "s": s,
            "l": l,
            "algorithms": ["EH17", "E20"]
          })
        })

def tabulate_ff_dh_schnorr(csv_prefix):
  with open(csv_prefix + "-params.csv", "w+") as file:
    csv_writer = csv.DictWriter(file, fieldnames=[
      "modulus_bitlength",
      "num_input_qubits",
      "num_shots",
      "pp_success_probability",
      "details"])
    csv_writer.writeheader()

    for [modulus_length, z, options] in ff_dh_schnorr_dlp_params:
      m = 2 * z

      # Single run.
      l = m
      sigma = 0
      num_input_qubits = m + sigma + l

      csv_writer.writerow({
        "modulus_bitlength": modulus_length,
        "num_input_qubits": num_input_qubits,
        "num_shots": 1,
        "pp_success_probability": 0.9999,
        "details": json.dumps({
          "z": z,
          "m": m,
          "sigma": sigma,
          "l": l,
          "algorithms": ["E19p"]
        })
      })

      # Multiple runs.
      for [s, n, sigma] in options:
        l = ceil(m / s)
        num_input_qubits = m + sigma + l

        csv_writer.writerow({
          "modulus_bitlength": modulus_length,
          "num_input_qubits": num_input_qubits,
          "num_shots": n,
          "pp_success_probability": 0.99,
          "details": json.dumps({
            "z": z,
            "m": m,
            "sigma": sigma,
            "s": s,
            "l": l,
            "algorithms": ["E19p"]
          })
        })


def tabulate_rsa(csv_prefix):
  with open(csv_prefix + "-params.csv", "w+") as file:
    csv_writer = csv.DictWriter(file, fieldnames=[
      "modulus_bitlength",
      "num_input_qubits",
      "num_shots",
      "pp_success_probability",
      "details"])
    csv_writer.writeheader()

    for [modulus_length, options] in rsa_params:
      if 0 != modulus_length % 2:
        raise Exception("Error: The modulus length must be even.")
      m = modulus_length // 2 - 1

      # Single run.
      for [delta, pp_tau, pp_t, pp_success_probability, pp_complexity] in \
        rsa_single_options:
        l = m - delta
        num_input_qubits = m + 2 * l

        csv_writer.writerow({
          "modulus_bitlength": modulus_length,
          "num_input_qubits": num_input_qubits,
          "num_shots": 1,
          "pp_success_probability": pp_success_probability,
          "details": json.dumps({
            "m": m,
            "delta": delta,
            "l": l,
            "pp_tau": pp_tau,
            "pp_t": pp_t,
            "pp_complexity": pp_complexity,
            "algorithms": ["EH17", "E23p"]
          })
        })

      # Multiple runs.
      for [s, n] in options:
        l = ceil(m / s)
        num_input_qubits = m + 2 * l

        csv_writer.writerow({
          "modulus_bitlength": modulus_length,
          "num_input_qubits": num_input_qubits,
          "num_shots": n,
          "pp_success_probability": 0.99,
          "details": json.dumps({
            "m": m,
            "s": s,
            "l": l,
            "algorithms": ["EH17", "E20"]
          })
        })

if __name__ == '__main__':
  tabulate_rsa("rsa")
  tabulate_ff_dh_short("ff-dh-short")
  tabulate_ff_dh_schnorr("ff-dh-schnorr")