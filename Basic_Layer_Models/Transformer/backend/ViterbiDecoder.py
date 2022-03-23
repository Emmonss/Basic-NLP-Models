
import numpy as np

def ViterbiDecoder(A,B,pi,O):
    T = len(O)
    N = len(A[0])

    delta = [[0]*N for _ in range(T)]
    psi = [[0]*N for _ in range(T)]

    #init
    for i in range(N):
        delta[0][i] = pi[i]*B[i][O[0]]
        psi[0][i] = 0

    #iter
    for t in range(1,T):
        for i in range(N):
            temp,maxindex = 0,0
            for j in range(N):
                res = delta[t-1][j]*A[j][i]
                if res > temp:
                    temp = res
                    maxindex = j
            delta[t][i] = temp*B[i][O[t]]
            psi[t][i] = maxindex

    #end
    p = max(delta[-1])
    for i in range(N):
        if [-1][i] ==p:
            i_T = i

    #backtrack
    path = [0]*T
    i_t = i_T
    for t in reversed(range(T-1)):
        i_t = psi[t+1][i_t]
        path[t] = i_t
    path[-1] = i_T

    return delta,psi,path

if __name__ == '__main__':
    A = [[0.5,0.2,0.3],
         [0.3,0.5,0.2],
         [0.2,0.3,0.5]]
    B = [[0.5,0.5],[0.4,0.6],[0.7,0.3]]
    pi = [0.2,0.4,0.4]
    O = [0,0,1,1,1,0,1,0,1,1,0]
    delta, psi, path = ViterbiDecoder(A,B,pi,O)
    print('-'*30)
    print(delta)
    print('-'*30)
    print(psi)
    print('-'*30)
    print(path)


