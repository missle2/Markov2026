import numpy as np
def P(m):
    A = np.zeros((m, m))
    for i in range(m-1): A[i, i+1] = 1
    A[m-1, 0] = A[m-1, 1] = .5
    return A
print('m', 'M', '(m-1)^2+1', '|l_2|')#, '1-|l_2|', '-1/ln|l_2|')
for m in range(3, 13):
    A = P(m); B = (A > 0).astype(int); M = 1
    while not B.all():
        B = ((B @ A) > 0).astype(int); M += 1
    l2 = sorted(abs(np.linalg.eigvals(A)))[-2]
    print(m, M, (m-1)**2 + 1, l2)
