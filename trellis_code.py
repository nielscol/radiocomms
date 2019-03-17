from scipy.stats import binom

GROUP_SIZE = 2
GROUPS = 7
BER = 1e-5

def detect_trellis_errors(seq):
    n_errors = 0
    for n in range(len(seq)-1):
        n_errors += seq[n][1]^seq[n+1][0]
    return n_errors

def detect_trellis_error_pos(seq):
    positions = [0]*(len(seq)-1)
    for n in range(len(seq)-1):
        positions[n] = seq[n][1]^seq[n+1][0]
    return positions


def trellis_test_1b_errors(seq):
    errors = []
    for m in range(len(seq)*GROUP_SIZE):
        test_seq = [list(group) for group in seq]
        test_seq[m/GROUP_SIZE][m%GROUP_SIZE] ^= 1
        errors.append(detect_trellis_errors(test_seq))
    return errors

def trellis_test_1b_error_pos(seq):
    error_pos = [0]*(len(seq)-1)
    for m in range(len(seq)*GROUP_SIZE):
        test_seq = [list(group) for group in seq]
        test_seq[m/GROUP_SIZE][m%GROUP_SIZE] ^= 1
        seq_e_pos = detect_trellis_error_pos(test_seq)
        for i, error in enumerate(seq_e_pos):
            error_pos[i] += error
    return error_pos

def trellis_test_2b_errors(seq):
    errors = []
    for m in range(len(seq)*GROUP_SIZE-1):
        for n in range(m+1,len(seq)*GROUP_SIZE):
            test_seq = [list(group) for group in seq]
            test_seq[m/GROUP_SIZE][m%GROUP_SIZE] ^= 1
            test_seq[n/GROUP_SIZE][n%GROUP_SIZE] ^= 1
            errors.append(detect_trellis_errors(test_seq))
    return errors

def trellis_test_2b_error_pos(seq):
    error_pos = [0]*(len(seq)-1)
    for m in range(len(seq)*GROUP_SIZE-1):
        for n in range(m+1,len(seq)*GROUP_SIZE):
            test_seq = [list(group) for group in seq]
            test_seq[m/GROUP_SIZE][m%GROUP_SIZE] ^= 1
            test_seq[n/GROUP_SIZE][n%GROUP_SIZE] ^= 1
            seq_e_pos = detect_trellis_error_pos(test_seq)
            for i, error in enumerate(seq_e_pos):
                error_pos[i] += error
    return error_pos

def seq_string(seq, group_size=GROUP_SIZE):
    text = ""
    for m, group in enumerate(seq):
        for n in range(group_size)[::-1]:
            text += "%d"%group[n]
        if m < len(seq)-1:
            text += ","
    return text

def gen_test_seqs(group_size=GROUP_SIZE, groups=GROUPS):
    seqs = []
    for m in range(2<<(group_size*groups-1)):
        curr_seq = []
        for n in range(groups):
            x = (m&2**(2*n)==2**(2*n), m&2**(2*n+1)==2**(2*n+1))
            curr_seq.append(x)
        seqs.append(tuple(curr_seq[::-1]))
        # print seq_string(curr_seq[::-1])
    return seqs

print("Running with %d groups x %d bits"%(GROUPS,GROUP_SIZE))

test_seqs = gen_test_seqs()

errors = []
valid_seqs = []
seq_by_n_errors = {}
#print("\nAll possible sequences")
for seq in test_seqs:
    errors.append(detect_trellis_errors(seq))
    # print("%d Errors\t%s"%(errors[-1], seq_string(seq)))
    if not errors[-1]:
        valid_seqs.append(seq)
    if errors[-1] not in seq_by_n_errors:
        seq_by_n_errors[errors[-1]] = []
    seq_by_n_errors[errors[-1]].append(seq)

print("\nAll valid sequences")
for n, seq in enumerate(valid_seqs):
    print("%d\t%s"%(n, seq_string(seq)))
print("\nCount of sequences by errors")
prob_of_n_errors = {}
for n in sorted(seq_by_n_errors):
    prob_of_n_errors[n] = binom.pmf(n=GROUP_SIZE*GROUPS, k=n, p=BER)
    print("%d Errors:\t%d\tprob = %E"%(n, len(seq_by_n_errors[n]),prob_of_n_errors[n]))

print("\n* Single bit error test")
error_test = []
error_positions = [0]*(GROUPS-1)
seqs = 0
detected_errors = 0
for seq in valid_seqs:
    error_test.append(trellis_test_1b_errors(seq))
    for i, error in enumerate(trellis_test_1b_error_pos(seq)):
        error_positions[i] += error
    seqs += len(error_test[-1])
    detected_errors += sum(error_test[-1])
print("Location of transition errors")
print("\t%r"%error_positions)
print("\n* Single bit error detection rate = %f"%(detected_errors/float(seqs)))


print("\n* Two bit error test")
error_test = []
error_positions = [0]*(GROUPS-1)
seqs = 0
detected_errors = 0
for seq in valid_seqs:
    error_test.append(trellis_test_2b_errors(seq))
    for i, error in enumerate(trellis_test_2b_error_pos(seq)):
        error_positions[i] += error
    #print error_test[-1]
print("Location of transition errors")
print("\t%r"%error_positions)



