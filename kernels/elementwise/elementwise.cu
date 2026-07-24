// 包含标准算法库（提供 std::min, std::max 等，虽然此处未显式使用）
#include <algorithm>
// CUDA bfloat16 数据类型支持（16 位浮点，与 FP16 指数位不同）
#include <cuda_bf16.h>
// CUDA half (FP16) 数据类型及内建函数（如 __hadd）
#include <cuda_fp16.h>
// CUDA FP8 数据类型支持（8 位浮点，本代码未使用但保留引入）
#include <cuda_fp8.h>
// CUDA 运行时 API（cudaMalloc, cudaLaunchKernel 等）
#include <cuda_runtime.h>
// 浮点数极限值定义（如 FLT_MAX, FLT_MIN）
#include <float.h>
// 标准输入输出（printf, fprintf 等）
#include <stdio.h>
// 标准库（malloc, free, exit 等）
#include <stdlib.h>
// PyTorch C++ 扩展的核心头文件（定义 torch::Tensor 等）
#include <torch/extension.h>
// PyTorch 张量类型定义
#include <torch/types.h>
// 标准向量容器（std::vector）
#include <vector>

// 定义 CUDA warp 的大小为 32 个线程（NVIDIA GPU 硬件固定）
#define WARP_SIZE 32
// 将任意变量地址强制转为 int4* 并解引用，用于 128 位加载/存储（本代码未使用）
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
// 将任意变量地址强制转为 float4* 并解引用，用于 128 位加载/存储
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
// 将任意变量地址强制转为 half2* 并解引用，用于 64 位加载/存储
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
// 将任意变量地址强制转为 __nv_bfloat162* 并解引用（bfloat16 双元，未使用）
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
// 将任意变量地址强制转为 float4* 并解引用，用于 128 位加载/存储（通用命名）
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// ---------- FP32 标量逐元素加法内核 ----------
// 网格大小：N/256，块大小：256，每个线程处理一个 float
__global__ void elementwise_add_f32_kernel(float *a, float *b, float *c,
                                           int N) {
  // 计算当前线程在整个网格中的全局线性索引（一维）
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // 防止越界，只处理有效索引
  if (idx < N)
    c[idx] = a[idx] + b[idx];  // 标量浮点加法
}

// ---------- FP32 向量化（Vec4）逐元素加法内核 ----------
// 每个线程处理 4 个 float（一个 float4），以提高内存带宽利用率
__global__ void elementwise_add_f32x4_kernel(float *a, float *b, float *c,
                                             int N) {
  // 当前线程负责的起始索引（每次跳过 4 个元素）
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  // 检查是否剩余至少 4 个连续元素
  if ((idx + 3) < N) {
    // 用 float4 一次性加载 4 个 float（128 位）
    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b = FLOAT4(b[idx]);
    float4 reg_c;
    // 分别相加四个分量，x,y,z,w是float4 4个成员对应名称
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    // 一次性写回 4 个 float（128 位）
    FLOAT4(c[idx]) = reg_c;
  } else if (idx < N) {
    // 剩余不足 4 个，用标量循环逐个处理
    for (int i = 0; (idx + i) < N; i++) {
      c[idx + i] = a[idx + i] + b[idx + i];
    }
  }
}

// ---------- FP16 标量逐元素加法内核 ----------
// 每个线程处理一个 half（16 位浮点）
__global__ void elementwise_add_f16_kernel(half *a, half *b, half *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    c[idx] = __hadd(a[idx], b[idx]);  // 使用半精度内建加法函数
}

// ---------- FP16 双元向量化（half2）内核 ----------
// 每个线程处理 2 个 half（一个 half2）
__global__ void elementwise_add_f16x2_kernel(half *a, half *b, half *c, int N) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  // 若至少剩余 2 个元素
  if ((idx + 1) < N) {
    // 加载两个 half2（分别来自 a 和 b）
    half2 reg_a = HALF2(a[idx]);
    half2 reg_b = HALF2(b[idx]);
    half2 reg_c;
    // 分别对两个分量做半精度加法
    reg_c.x = __hadd(reg_a.x, reg_b.x);
    reg_c.y = __hadd(reg_a.y, reg_b.y);
    // 写回一个 half2
    HALF2(c[idx]) = reg_c;
  } else if (idx < N) {
    // 剩余 1 个，标量处理
    c[idx] = __hadd(a[idx], b[idx]);
  }
}

// ---------- FP16 8 元素显式展开内核 ----------
// 每个线程处理 8 个 half（即 4 个 half2），将循环手动展开
__global__ void elementwise_add_f16x8_kernel(half *a, half *b, half *c, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if ((idx + 7) < N) {
    // 从 a 加载 4 个 half2（对应 a[idx+0] ~ a[idx+7]）
    half2 reg_a_0 = HALF2(a[idx + 0]);
    half2 reg_a_1 = HALF2(a[idx + 2]);
    half2 reg_a_2 = HALF2(a[idx + 4]);
    half2 reg_a_3 = HALF2(a[idx + 6]);
    // 从 b 加载 4 个 half2
    half2 reg_b_0 = HALF2(b[idx + 0]);
    half2 reg_b_1 = HALF2(b[idx + 2]);
    half2 reg_b_2 = HALF2(b[idx + 4]);
    half2 reg_b_3 = HALF2(b[idx + 6]);
    // 声明结果寄存器（4 个 half2）
    half2 reg_c_0, reg_c_1, reg_c_2, reg_c_3;
    // 分别对每对 half2 的两个分量做加法
    reg_c_0.x = __hadd(reg_a_0.x, reg_b_0.x);
    reg_c_0.y = __hadd(reg_a_0.y, reg_b_0.y);
    reg_c_1.x = __hadd(reg_a_1.x, reg_b_1.x);
    reg_c_1.y = __hadd(reg_a_1.y, reg_b_1.y);
    reg_c_2.x = __hadd(reg_a_2.x, reg_b_2.x);
    reg_c_2.y = __hadd(reg_a_2.y, reg_b_2.y);
    reg_c_3.x = __hadd(reg_a_3.x, reg_b_3.x);
    reg_c_3.y = __hadd(reg_a_3.y, reg_b_3.y);
    // 将 4 个 half2 结果写回 c
    HALF2(c[idx + 0]) = reg_c_0;
    HALF2(c[idx + 2]) = reg_c_1;
    HALF2(c[idx + 4]) = reg_c_2;
    HALF2(c[idx + 6]) = reg_c_3;
  } else if (idx < N) {
    // 剩余不足 8 个，标量循环处理
    for (int i = 0; (idx + i) < N; i++) {
      c[idx + i] = __hadd(a[idx + i], b[idx + i]);
    }
  }
}

// ---------- FP16 打包 8 元素内核（优化版） ----------
// 使用局部数组 + 128 位一次性加载/存储，进一步减少内存事务
__global__ void elementwise_add_f16x8_pack_kernel(half *a, half *b, half *c,
                                                  int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  if ((idx + 7) < N) {
    // 在寄存器（或 .local 空间）声明 8 个 half 的数组（共 128 位）
    half pack_a[8], pack_b[8], pack_c[8];
    // 将 pack_a[0] 的首地址视为 float4*，一次性加载 128 位（8 个 half）
    LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
    // 同样加载 b
    LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);

    // 循环展开，每轮处理一个 half2（两个 half）
    #pragma unroll
    for (int i = 0; i < 8; i += 2) {
      // __hadd2 直接对两个 half2 做加法，返回 half2
      HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
    }
    // 将结果数组的首地址视为 float4*，一次性写回 128 位
    LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
  } else if (idx < N) {
    // 边界剩余元素逐个处理
    for (int i = 0; (idx + i) < N; i++) {
      c[idx + i] = __hadd(a[idx + i], b[idx + i]);
    }
  }
}

// ---------- 以下为 PyTorch 绑定相关宏 ----------

// 将宏参数转换为字符串（用于函数名注册）
#define STRINGFY(str) #str

// 通用绑定宏：将函数 func 注册到模块 m，Python 中函数名与 C++ 函数名相同
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

// 检查张量数据类型是否等于预期类型，否则输出信息并抛异常
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

// 主要宏：生成一个完整的 PyTorch 扩展函数，名为 elementwise_add_<packed_type>
// 参数：
//   packed_type  : 内核名称后缀（如 f32, f16x8）
//   th_type      : PyTorch 数据类型（如 torch::kFloat32）
//   element_type : C++ 数据类型（如 float, half）
//   n_elements   : 每个线程处理的元素个数（1, 2, 4, 8）
#define TORCH_BINDING_ELEM_ADD(packed_type, th_type, element_type, n_elements) \
  void elementwise_add_##packed_type(torch::Tensor a, torch::Tensor b,         \
                                     torch::Tensor c) {                        \
    /* 检查三个张量的数据类型 */                                               \
    CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(b, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(c, (th_type))                                     \
    /* 获取张量 a 的维度数 */                                                  \
    const int ndim = a.dim();                                                  \
    /* 如果维度不是 2（即非矩阵），采用一维扁平化调度 */                       \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= a.size(i);  /* 各维度大小相乘得到总元素数 */                     \
      }                                                                        \
      /* 块大小 = 256 / n_elements，保证总线程数 = 256 */                     \
      dim3 block(256 / (n_elements));                                          \
      /* 网格大小向上取整，覆盖所有元素 */                                     \
      dim3 grid((N + 256 - 1) / 256);                                          \
      /* 启动内核，传入设备指针（强转为 element_type*） */                    \
      elementwise_add_##packed_type##_kernel<<<grid, block>>>(                 \
          reinterpret_cast<element_type *>(a.data_ptr()),                      \
          reinterpret_cast<element_type *>(b.data_ptr()),                      \
          reinterpret_cast<element_type *>(c.data_ptr()), N);                  \
    } else {                                                                   \
      /* 张量为 2 维（矩阵 [S, K]），可进行按行优化 */                         \
      const int S = a.size(0);  /* 行数 */                                     \
      const int K = a.size(1);  /* 列数 */                                     \
      const int N = S * K;      /* 总元素数 */                                 \
      /* 如果每行的向量化块数（K / n_elements）不超过 1024（最大线程块大小） */ \
      if ((K / (n_elements)) <= 1024) {                                        \
        /* 设置块大小为每行的块数，网格大小为行数，实现“一行一线程块” */      \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        elementwise_add_##packed_type##_kernel<<<grid, block>>>(               \
            reinterpret_cast<element_type *>(a.data_ptr()),                    \
            reinterpret_cast<element_type *>(b.data_ptr()),                    \
            reinterpret_cast<element_type *>(c.data_ptr()), N);                \
      } else {                                                                 \
        /* 列数过大，每行块数超过 1024，退回到一维扁平化调度 */                \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= a.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        elementwise_add_##packed_type##_kernel<<<grid, block>>>(               \
            reinterpret_cast<element_type *>(a.data_ptr()),                    \
            reinterpret_cast<element_type *>(b.data_ptr()),                    \
            reinterpret_cast<element_type *>(c.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }

// 使用宏生成 6 个具体的数据类型/向量化版本
TORCH_BINDING_ELEM_ADD(f32, torch::kFloat32, float, 1)     // FP32 标量
TORCH_BINDING_ELEM_ADD(f32x4, torch::kFloat32, float, 4)   // FP32 向量化 4
TORCH_BINDING_ELEM_ADD(f16, torch::kHalf, half, 1)         // FP16 标量
TORCH_BINDING_ELEM_ADD(f16x2, torch::kHalf, half, 2)       // FP16 half2
TORCH_BINDING_ELEM_ADD(f16x8, torch::kHalf, half, 8)       // FP16 显式 8 元素
TORCH_BINDING_ELEM_ADD(f16x8_pack, torch::kHalf, half, 8)  // FP16 打包 8 元素

// 定义 PyTorch 扩展模块的入口，绑定所有函数
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8_pack)
}
