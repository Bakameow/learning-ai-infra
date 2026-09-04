# Softmax
## 
```markdown
softmax = e^x / \sum{e^x}
```
1. 为了避免指数溢出，需要使用safe softmax，每行各个元素减去对应行的最大值，softmax的结果仍保持稳定

## 优化思路
1. 行并行。起一个线程块，每个线程负责一行的softmax运算
2. 行内并行，行内多个线程使用shared memory保存各自部分的最大值，之后在shared memory上使用reduce计算得到行内的最大值。

## 线程束洗牌指令
1. __shfl_sync
1. __shfl_up_sync：数据值向上（坐标大的线程）平移
1. __shfl_down_sync：数据向下（坐标小的线程）平移
```c++
__global__ void softmax_forward_kernel2(float *out, const float *inp, int N, int C){
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    const float* x = inp + idx * C;
    float max = -inf;
    for(int i=tid;i<C;i+=block_size){
        max = fmaxf(x[i],max);
    }
    shared[tid]=max;
    __syncthreads();
    for(int stride=block_size/2;stride>0;stride/=2){
        if(tid<stride)
            shared[tid] = fmaxf(shared[tid],shared[tid+stride]);
        __syncthreads();
    }
    float offset = shared[0];

    for(int i=tid;i<C;i+=block_size){
        out[idx * C + i] = exp(x[i] - Q：offset);
    }
    x = out + idx * C;
    float sumval = 0;
    for(int i=tid;i<C;i+=block_size){
        sumval+=x[i];
    }
    shared[tid]=sumval;
    __syncthreads();
    for(int stride=block_size;stride>=1;stride/=2){
        if(tid<stride)shared[tid]+=shared[tid+stride];
        __sync();
    }
    __sync();
    float sum=shared[0];
    

}
```