import numpy as np
import pandas as pd
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import threading
import pycuda.gpuarray as gpuarray

# BLOCK_SIZE = 16
BLOCK_SIZE = 32



# Use Gauss Elimination for Ax = b
def solve_cuda64(A, b):
    n = len(A)
    ni = np.int32(n)
    Af = A.astype(np.float64)
    bf = b.astype(np.float64)
    
    x = np.empty([n]).astype(np.float64)

    # allocate memory on device
    A_gpu = cuda.mem_alloc(Af.nbytes)
    b_gpu = cuda.mem_alloc(bf.nbytes)
    x_gpu = cuda.mem_alloc(x.nbytes)


    # copy matrix to memory
    cuda.memcpy_htod(A_gpu, Af)
    cuda.memcpy_htod(b_gpu, bf)

    # compile kernel
    mod = SourceModule(open("kernels.cu", "r").read())

    # get function
    linear_solve_one_column = mod.get_function("linear_solve_one_column");
    linear_solve_clear_one_column = mod.get_function("linear_solve_clear_one_column");

    # set grid size
    n1 = n
    n2 = n + 1
    if n2%BLOCK_SIZE != 0 and n1%BLOCK_SIZE != 0:
        grid=(n2//BLOCK_SIZE+1,n1//BLOCK_SIZE+1,1)
    elif n2%BLOCK_SIZE == 0 and n1%BLOCK_SIZE != 0:
        grid=(n2//BLOCK_SIZE,n1//BLOCK_SIZE+1,1)
    elif n2%BLOCK_SIZE != 0 and n1%BLOCK_SIZE == 0:
        grid=(n2//BLOCK_SIZE+1,n1//BLOCK_SIZE,1)
    else:
        grid=(n2//BLOCK_SIZE,n1//BLOCK_SIZE,1)

    # call gpu function
    for icol in range(n):
        icoli = np.int32(icol)
        linear_solve_one_column(icoli, ni, A_gpu, b_gpu, block=(BLOCK_SIZE,BLOCK_SIZE,1), grid=grid)
        cuda.Context.synchronize()
    linear_solve_clear_one_column(x_gpu, icoli, ni, A_gpu, b_gpu, block=(BLOCK_SIZE,BLOCK_SIZE,1), grid=grid)
    cuda.Context.synchronize()

    # copy back the result
    cuda.memcpy_dtoh(x, x_gpu)

    return x


# cuda blocks and grids are set to cover all indexes. So there are surplus processes that should not be performed. This function confirms that there are enough processes that cover all necessary indexes. 
def cuda_cover_test():
    n = 35
    ni = np.int32(n)
    x = np.empty([n,n]).astype(np.float32)

    # allocate memory on device
    x_gpu = cuda.mem_alloc(x.nbytes)

    # compile kernel
    mod = SourceModule(open("kernels.cu", "r").read())

    # get function
    cuda_cover_test = mod.get_function("cuda_cover_test");

    # set grid size
    n1 = 10
    n2 = 15
    if n2%BLOCK_SIZE != 0 and n1%BLOCK_SIZE != 0:
        grid=(n2//BLOCK_SIZE+1,n1//BLOCK_SIZE+1,1)
    elif n2%BLOCK_SIZE == 0 and n1%BLOCK_SIZE != 0:
        grid=(n2//BLOCK_SIZE,n1//BLOCK_SIZE+1,1)
    elif n2%BLOCK_SIZE != 0 and n1%BLOCK_SIZE == 0:
        grid=(n2//BLOCK_SIZE+1,n1//BLOCK_SIZE,1)
    else:
        grid=(n2//BLOCK_SIZE,n1//BLOCK_SIZE,1)

    # print('grid:', grid)   # must be integers

    # call gpu function
    cuda_cover_test(x_gpu, ni, block=(BLOCK_SIZE,BLOCK_SIZE,1), grid=grid)
    cuda.Context.synchronize()

    # copy back the result
    cuda.memcpy_dtoh(x, x_gpu)

#     print(x)
    return x


def logGaussians(data_in, mean, Cov):
    ndata, dim = data_in.shape
    ndatai = np.int32(ndata)
    dimi = np.int32(dim)
    dataf = data_in.astype(np.float32)
    meanf = mean.astype(np.float32)
    invCov = np.linalg.inv(Cov)
    invCovf = invCov.astype(np.float32)
    detCov = np.linalg.det(Cov)
    detCovf = np.float32(detCov)

    lnPs_out = np.empty(ndata).astype(np.float32)

    # allocate memory on device
    data_gpu = cuda.mem_alloc(dataf.nbytes)
    mean_gpu = cuda.mem_alloc(meanf.nbytes)
    invCov_gpu = cuda.mem_alloc(invCovf.nbytes)
    lnPs_gpu = cuda.mem_alloc(lnPs_out.nbytes)

    # copy matrix to memory
    cuda.memcpy_htod(data_gpu, dataf)
    cuda.memcpy_htod(mean_gpu, meanf)
    cuda.memcpy_htod(invCov_gpu, invCovf)

    # compile kernel
    mod = SourceModule(open("kernels.cu", "r").read())

    # get function
    logGaussian = mod.get_function("logGaussian");

    # set grid size
    if ndata%BLOCK_SIZE != 0:
        grid=(ndata//BLOCK_SIZE+1,1,1)
    else:
        grid=(ndata//BLOCK_SIZE,1,1)

    logGaussian(lnPs_gpu, ndatai, dimi, data_gpu, mean_gpu, invCov_gpu, detCovf, block=(BLOCK_SIZE,1,1), grid=grid)

    # copy back the result
    cuda.memcpy_dtoh(lnPs_out, lnPs_gpu)
    return lnPs_out





def calculateDist_cuda(data1, data2):
    n1, dim = data1.shape
    n2 = data2.shape[0]
    # print(n1, n2, dim)
    n1i = np.int32(n1)
    n2i = np.int32(n2)
    dimi = np.int32(dim)
    data1f = data1.astype(np.float32)
    data2f = data2.astype(np.float32)

    # distance matrix
    DistSqMat = np.empty([n1, n2]).astype(np.float32)
    # DistSqMat = np.empty([n1, n2]).astype(np.float64)
    # print(DistSqMat)

    # allocate memory on device
    data1_gpu = cuda.mem_alloc(data1f.nbytes)
    data2_gpu = cuda.mem_alloc(data2f.nbytes)
    DistSqMat_gpu = cuda.mem_alloc(DistSqMat.nbytes)

    # copy matrix to memory
    cuda.memcpy_htod(data1_gpu, data1f)
    cuda.memcpy_htod(data2_gpu, data2f)

    # compile kernel
    mod = SourceModule(open("kernels.cu", "r").read())

    # get function
    distCal = mod.get_function("distCal");

    # set grid size
    if n2%BLOCK_SIZE != 0 and n1%BLOCK_SIZE != 0:
        grid=(n2//BLOCK_SIZE+1,n1//BLOCK_SIZE+1,1)
    elif n2%BLOCK_SIZE == 0 and n1%BLOCK_SIZE != 0:
        grid=(n2//BLOCK_SIZE,n1//BLOCK_SIZE+1,1)
    elif n2%BLOCK_SIZE != 0 and n1%BLOCK_SIZE == 0:
        grid=(n2//BLOCK_SIZE+1,n1//BLOCK_SIZE,1)
    else:
        grid=(n2//BLOCK_SIZE,n1//BLOCK_SIZE,1)

    # print('grid:', grid)   # must be integers

    # call gpu function
    distCal(DistSqMat_gpu, n1i, n2i, dimi, data1_gpu, data2_gpu, block=(BLOCK_SIZE,BLOCK_SIZE,1), grid=grid)

    # copy back the result
    cuda.memcpy_dtoh(DistSqMat, DistSqMat_gpu)

    # print(DistSqMat)

    return DistSqMat




def calculateKDEs_cuda(tstPts, Data, h_bw=1.):
    n1, dim = tstPts.shape
    n2 = Data.shape[0]
    # print(n1, n2, dim)
    n1i = np.int32(n1)
    n2i = np.int32(n2)
    dimi = np.int32(dim)
    data1f = tstPts.astype(np.float32)
    data2f = Data.astype(np.float32)
    h_bwf = np.float32(h_bw)

    # distance matrix
    Kmat = np.empty([n1, n2]).astype(np.float32)
    # Kmat = np.empty([n1, n2]).astype(np.float64)
    # print(Kmat)

    # allocate memory on device
    data1_gpu = cuda.mem_alloc(data1f.nbytes)
    data2_gpu = cuda.mem_alloc(data2f.nbytes)
    Kmat_gpu = cuda.mem_alloc(Kmat.nbytes)

    # copy matrix to memory
    cuda.memcpy_htod(data1_gpu, data1f)
    cuda.memcpy_htod(data2_gpu, data2f)

    # compile kernel
    mod = SourceModule(open("kernels.cu", "r").read())

    # get function
    kernelvals = mod.get_function("kernelvals");

    # set grid size
    if n2%BLOCK_SIZE != 0 and n1%BLOCK_SIZE != 0:
        grid=(n2//BLOCK_SIZE+1,n1//BLOCK_SIZE+1,1)
    elif n2%BLOCK_SIZE == 0 and n1%BLOCK_SIZE != 0:
        grid=(n2//BLOCK_SIZE,n1//BLOCK_SIZE+1,1)
    elif n2%BLOCK_SIZE != 0 and n1%BLOCK_SIZE == 0:
        grid=(n2//BLOCK_SIZE+1,n1//BLOCK_SIZE,1)
    else:
        grid=(n2//BLOCK_SIZE,n1//BLOCK_SIZE,1)

    # print('grid:', grid)   # must be integers

    # call gpu function
    kernelvals(Kmat_gpu, n1i, n2i, dimi, data1_gpu, data2_gpu, h_bwf, block=(BLOCK_SIZE,BLOCK_SIZE,1), grid=grid)

    # copy back the result
    cuda.memcpy_dtoh(Kmat, Kmat_gpu)

    # print(Kmat)

    return Kmat

\
if __name__ == "__main__":
    tstn1 = 32
    trn = 3000000
    dim = 5
    mean1 = np.zeros(dim)
    mean1[0] = 1
    Cov1 = np.eye(dim)
    # mean2 = np.zeros([1,dim])
    mean2 = np.zeros(dim)
    mean2[0] = -1
    Cov2 = np.eye(dim)
    tstX1 = np.random.multivariate_normal(mean1, Cov1, tstn1)
    totX = np.random.multivariate_normal(mean2, Cov2, trn)

    print('Data generation complete\n')

    invCov = np.eye(dim)
    mu_mean = np.zeros([1,2])


    tstPt = np.array([-4, 0]).reshape([-1,2])
    print(tstPt)
    tstmean = np.array([-2, 0]).reshape([-1,2])
    tstCov = np.array([[1,0],[0,1]])
    logPs = logGaussians(tstPt, tstmean, tstCov)
    print(logPs)




