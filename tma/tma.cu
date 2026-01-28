#include <chrono>
#include <cstdio>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda/std/cmath>
#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>

template <typename T> struct defer {
  T op_;
  defer(T &&op) : op_{std::move(op)} {}
  defer(defer const &) = delete;
  defer(defer &&) = delete;
  defer &operator=(defer const &) = delete;
  defer &operator=(defer &&) = delete;
  ~defer() { op_(); }
};

using i32 = int32_t;
using i64 = int64_t;
using u32 = uint32_t;
using u64 = uint64_t;
using f32 = float;
using f64 = double;

using DType = f32;

template <typename T> defer(T) -> defer<T>;

#define GENERIC_OP(a, b, c)                                                    \
  cuda::std::fma(cuda::std::fma(a, b, c), cuda::std::fma(c, a, b),             \
                 cuda::std::fma(b, c, a))

// ----------------------
// Kernel 1: normal grid-strided FMA
// ----------------------
template <i32 ARITHMETIC_INTENSITY>
__global__ void fma_no_tma(DType const *__restrict__ src1,
                           DType const *__restrict__ src2,
                           DType *__restrict__ dst1, i64 n) {
  i64 idx = (i64)blockIdx.x * (i64)blockDim.x + (i64)threadIdx.x;
  i64 stride = (i64)blockDim.x * (i64)gridDim.x;

  for (i64 i = idx; i < n; i += stride) {
    DType a = src1[i];
    DType b = src2[i];
    DType c = a + b;
#pragma unroll
    for (i32 k = 0; k < ARITHMETIC_INTENSITY; ++k) {
      c = GENERIC_OP(a, b, c);
    }
    dst1[i] = c;
  }
}

template <i32 ARITHMETIC_INTENSITY>
__global__ void fma_no_tma_no_restrict(DType const *src1, DType const *src2,
                                       DType *dst1, i64 n) {
  i64 idx = (i64)blockIdx.x * (i64)blockDim.x + (i64)threadIdx.x;
  i64 stride = (i64)blockDim.x * (i64)gridDim.x;

  for (i64 i = idx; i < n; i += stride) {
    DType a = src1[i];
    DType b = src2[i];
    DType c = a + b;
#pragma unroll
    for (i32 k = 0; k < ARITHMETIC_INTENSITY; ++k) {
      c = GENERIC_OP(a, b, c);
    }
    dst1[i] = c;
  }
}

constexpr i32 BULK_COPY_ALIGNMENT = 128;
constexpr i32 BULK_COPY_SIZE_MULTIPLE = 16;

// ----------------------
// Kernel 2: real TMA (SM90+) FMA
// ----------------------

__device__ inline void tile_load2(u64 *__restrict__ bar,
                                  void const *__restrict__ src1,
                                  void const *__restrict__ src2,
                                  void *__restrict__ dst1,
                                  void *__restrict__ dst2, u32 tile_bytes) {
  u32 copied = 0;

  cuda::ptx::cp_async_bulk(cuda::ptx::space_shared, cuda::ptx::space_global,
                           dst1, src1, tile_bytes, bar);
  copied += tile_bytes;
  cuda::ptx::cp_async_bulk(cuda::ptx::space_shared, cuda::ptx::space_global,
                           dst2, src2, tile_bytes, bar);
  copied += tile_bytes;

  cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                       cuda::ptx::scope_cta,
                                       cuda::ptx::space_shared, bar, copied);
}

template <i32 ARITHMETIC_INTENSITY>
__device__ inline void process_tile2(DType const *__restrict__ tile1,
                                     DType const *__restrict__ tile2,
                                     i32 tile_offset, i32 num_tile_elements,
                                     DType *__restrict__ out, i32 stride) {
  for (i64 i = tile_offset; i < num_tile_elements; i += stride) {
    DType a = tile1[i];
    DType b = tile2[i];
    DType c = a + b;
#pragma unroll
    for (i32 k = 0; k < ARITHMETIC_INTENSITY; ++k) {
      c = GENERIC_OP(a, b, c);
    }
    out[i] = c;
  }
}

__device__ inline void wait_tma_parity(u64 *__restrict__ bar, u32 parity) {
  do {
  } while (!cuda::ptx::mbarrier_try_wait_parity(bar, parity));
}

__device__ inline static bool elect_one() {
  u32 const membermask = ~0;
  u32 is_elected;
  asm volatile("{\n\t .reg .pred P_OUT; \n\t"
               "elect.sync _|P_OUT, %1;\n\t"
               "selp.b32 %0, 1, 0, P_OUT; \n"
               "}"
               : "=r"(is_elected)
               : "r"(membermask)
               :);
  return threadIdx.x < 32 && static_cast<bool>(is_elected);
}

template <i32 TILE_DIM, i32 ARITHMETIC_INTENSITY>
__global__ void fma_tma_grid_strided(DType const *__restrict__ src1,
                                     DType const *__restrict__ src2,
                                     DType *__restrict__ dst1, i64 n) {
  __shared__ u64 bar;
  __shared__ alignas(BULK_COPY_ALIGNMENT) DType tile1[TILE_DIM];
  __shared__ alignas(BULK_COPY_ALIGNMENT) DType tile2[TILE_DIM];

  // each block processes a tile and strides over the grid dimension to get the
  // next tile
  i64 const stride = (i64)gridDim.x * TILE_DIM;

  // Initialize barrier once
  if (threadIdx.x == 0) {
    cuda::ptx::mbarrier_init(&bar, 1);
  }
  __syncthreads();

  u32 wait_parity = 0;

  for (i64 tile_begin = (i64)blockIdx.x * TILE_DIM; tile_begin < n;
       tile_begin += stride, wait_parity ^= 1) {
    i64 const tile_end = min(tile_begin + TILE_DIM, n);
    i64 const num_tile_elements = tile_end - tile_begin;
    u32 const tile_bytes = num_tile_elements * sizeof(DType);

    // Issue TMA copy into shared memory
    if (elect_one()) {
      tile_load2(&bar, src1 + tile_begin, src2 + tile_begin, tile1, tile2,
                 TILE_DIM * sizeof(DType));
    }

    // wait for TMA
    __syncthreads();
    wait_tma_parity(&bar, wait_parity);

    i64 const tile_stride = blockDim.x;
    process_tile2<ARITHMETIC_INTENSITY>(tile1, tile2, threadIdx.x,
                                        num_tile_elements, dst1 + tile_begin,
                                        blockDim.x);
  }
}

// TODO: implement
template <i32 TILE_DIM, i32 ARITHMETIC_INTENSITY>
__global__ void fma_tma_shmem(DType const *__restrict__ src1,
                              DType const *__restrict__ src2,
                              DType *__restrict__ dst1, i64 n) {
  __shared__ u64 bar;
  __shared__ alignas(BULK_COPY_ALIGNMENT) DType tile1[TILE_DIM];
  __shared__ alignas(BULK_COPY_ALIGNMENT) DType tile2[TILE_DIM];

  static_assert((TILE_DIM * sizeof(DType)) % BULK_COPY_SIZE_MULTIPLE == 0,
                "TILE_DIM must be multiple of bulk copy size");

  // Initialize barrier once
  if (threadIdx.x == 0) {
    cuda::ptx::mbarrier_init(&bar, 1);
  }
  __syncthreads();

  i64 const tile_begin = (i64)blockIdx.x * TILE_DIM;
  i64 const tile_end = min(tile_begin + TILE_DIM, n);
  i32 const num_tile_elements = (i32)(tile_end - tile_begin);

  // Issue TMA copy into shared memory
  if (elect_one()) {
    tile_load2(&bar, src1 + tile_begin, src2 + tile_begin, tile1, tile2,
               TILE_DIM * sizeof(DType));
  }

  // wait for TMA
  __syncthreads();
  u32 const wait_parity = 0;
  wait_tma_parity(&bar, wait_parity);

  process_tile2<ARITHMETIC_INTENSITY>(tile1, tile2, threadIdx.x,
                                      num_tile_elements, dst1 + tile_begin,
                                      blockDim.x);
}

template <i32 ARITHMETIC_INTENSITY>
__global__ void fma_tma_tuned_dyn_shmem(DType const *__restrict__ src1,
                                        DType const *__restrict__ src2,
                                        i32 bar_offset, i32 tile1_offset,
                                        i32 tile2_offset, i32 tile_dim,
                                        DType *__restrict__ dst1, i64 n) {
  extern __shared__ char smem[];
  auto *__restrict__ bar = reinterpret_cast<u64 *>(smem + bar_offset);
  auto *__restrict__ tile1 = reinterpret_cast<DType *>(smem + tile1_offset);
  auto *__restrict__ tile2 = reinterpret_cast<DType *>(smem + tile2_offset);

  //   static_assert(tile_dim % BULK_COPY_SIZE_MULTIPLE == 0,
  // "TILE_DIM must be multiple of bulk copy size");

  // Initialize barrier once
  if (threadIdx.x == 0) {
    cuda::ptx::mbarrier_init(bar, 1);
  }
  __syncthreads();

  i64 const tile_begin = (i64)blockIdx.x * tile_dim;
  i64 const tile_end = min(tile_begin + tile_dim, n);
  i32 const num_tile_elements = (i32)(tile_end - tile_begin);

  // Issue TMA copy into shared memory
  if (elect_one()) {
    tile_load2(bar, src1 + tile_begin, src2 + tile_begin, tile1, tile2,
               tile_dim * sizeof(DType));
  }

  // wait for TMA
  __syncthreads();
  u32 const wait_parity = 0;
  wait_tma_parity(bar, wait_parity);

  process_tile2<ARITHMETIC_INTENSITY>(tile1, tile2, threadIdx.x,
                                      num_tile_elements, dst1 + tile_begin,
                                      blockDim.x);
}

#define LOG(...)                                                               \
  do {                                                                         \
    printf(__VA_ARGS__);                                                       \
    fflush(stdout);                                                            \
  } while (0)

#define CHECK_CUDA_ERR()                                                       \
  do {                                                                         \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      LOG("CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__,      \
          __LINE__);                                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define TIMEIT(name, ...)                                                      \
  do {                                                                         \
    LOG("Starting %s...\n", name);                                             \
    auto start = std::chrono::high_resolution_clock::now();                    \
    __VA_ARGS__;                                                               \
    auto stop = std::chrono::high_resolution_clock::now();                     \
    f32 ms = std::chrono::duration<f32, std::milli>(stop - start).count();     \
    LOG("Finished %s took %.8f ms\n", name, ms);                               \
  } while (0)

i32 block_smem_capacity() {
  i32 max_smem;
  i32 device = 0;
  cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlock, device);
  CHECK_CUDA_ERR();
  return max_smem;
}

constexpr u64 alignup(u64 value, u64 alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

constexpr auto kernel_smem_requirement(i32 tile_dim) {
  constexpr auto loaded_bytes_per_iter = sizeof(DType) * 2; // tile1 and tile2
  auto bar_size = alignup(i32{sizeof(u64)}, BULK_COPY_ALIGNMENT); // bar
  auto tile_mem_size =
      alignup(tile_dim * loaded_bytes_per_iter, BULK_COPY_ALIGNMENT);
  return bar_size + tile_mem_size;
}

template <typename KernelFn>
i32 max_sm_occupancy(KernelFn kernel, i32 block_size, i32 dynamic_smem_bytes) {
  // Older drivers have issues handling CUkernel in the occupancy queries, get
  // the CUfunction instead.
  i32 max_occupancy = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_occupancy, kernel,
                                                block_size, dynamic_smem_bytes);
  CHECK_CUDA_ERR();
  return max_occupancy;
}

// Returns the minimum bytes in flight for a given architecture
// needed to saturate the TMA pipes
constexpr i32 min_saturating_bytes_in_flight(i32 sm_arch) {
  if (sm_arch >= 900) {
    // 32 for H100, 48 for H200
    return 48 * 1024;
  }

  if (sm_arch >= 800) {
    // A100
    return 16 * 1024;
  }

  // V100 and below
  return 12 * 1024;
}

struct saturation_stats {
  i32 tile_dim = 0; ///< Tile dimension that can achieve saturation
  i32 smem_req = 0; ///< Shared memory requirement for the tile dimension
};

struct tma_policy {
  /////////// tuning parameters
  // 256 on sm_90, 128 on sm_100
  static constexpr i32 block_dim = 256;
  static constexpr i32 min_tile_dim = 1 * block_dim;
  static constexpr i32 max_tile_dim = 65536 * block_dim;
};

template <typename KernelFn>
saturation_stats tune_tma_kernel(KernelFn kernel_fn) {

  // assuming sm_90+
  constexpr auto sat_bytes_in_flight = min_saturating_bytes_in_flight(900);
  auto const max_smem = block_smem_capacity();

  ///////////// kernel-derived parameters
  // loading 2 arrays
  constexpr i32 loaded_bytes_per_iter = sizeof(DType) * 2;

  // ensures the loop below runs at least once
  static_assert(tma_policy::min_tile_dim <= tma_policy::max_tile_dim, "");

  saturation_stats last_counts{};

  // Increase the number of elements loaded into the tiles until we saturate
  // the TMA pipes or exceed shared memory
  for (i32 tile_dim = tma_policy::min_tile_dim;
       tile_dim < tma_policy::max_tile_dim; tile_dim += tma_policy::block_dim) {
    i32 const smem_req = kernel_smem_requirement(tile_dim);
    assert(!(tile_dim == tma_policy::min_tile_dim && smem_req > max_smem) &&
           "min_tile_dim exceeds available shared memory");

    if (smem_req > max_smem) {
      break;
    }

    i32 const max_occupancy =
        max_sm_occupancy(kernel_fn, tma_policy::block_dim, smem_req);
    i32 const bytes_in_flight_SM =
        max_occupancy * tile_dim * loaded_bytes_per_iter;

    if (bytes_in_flight_SM >= sat_bytes_in_flight) {
      last_counts = {tile_dim, smem_req};
      break;
    }

    last_counts = {tile_dim, smem_req};
  }

  return last_counts;
}

void generate_input(DType *&dA, DType *&dB, i64 N) {
  cudaMalloc(&dA, N * sizeof(DType));
  CHECK_CUDA_ERR();
  cudaMalloc(&dB, N * sizeof(DType));
  CHECK_CUDA_ERR();

  DType *hA = new DType[N];
  defer hA_{[&] { delete[] hA; }};
  DType *hB = new DType[N];
  defer hB_{[&] { delete[] hB; }};
  for (i64 i = 0; i < N; ++i) {
    hA[i] = static_cast<DType>(i % 100) / 100.0f;
    hB[i] = static_cast<DType>((i + 37) % 100) / 100.0f;
  }

  cudaMemcpy(dA, hA, N * sizeof(DType), cudaMemcpyHostToDevice);
  CHECK_CUDA_ERR();
  cudaMemcpy(dB, hB, N * sizeof(DType), cudaMemcpyHostToDevice);
  CHECK_CUDA_ERR();
}

void log_diff(const char *label, DType *a, DType *b, i64 N) {
  DType *ha = new DType[N];
  defer ha_{[&] { delete[] ha; }};
  cudaMemcpy(ha, a, N * sizeof(DType), cudaMemcpyDeviceToHost);
  CHECK_CUDA_ERR();
  DType *hb = new DType[N];
  defer hb_{[&] { delete[] hb; }};
  cudaMemcpy(hb, b, N * sizeof(DType), cudaMemcpyDeviceToHost);
  CHECK_CUDA_ERR();

  DType diff = 0.0f;

  for (i64 i = 0; i < N; ++i) {
    diff += std::abs(ha[i] - hb[i]);
  }

  LOG("Total difference (%s): %e\n", label, diff);
}

#define BENCHMARK_PRELUDE                                                      \
  static constexpr i32 ARITHMETIC_INTENSITY = 32;                              \
  i64 const N = state.get_int64("N");                                          \
  constexpr i32 TILE_DIM = 4096;                                               \
                                                                               \
  DType *dA;                                                                   \
  DType *dB;                                                                   \
  generate_input(dA, dB, N);                                                   \
  defer dA_{[&] { cudaFree(dA); }};                                            \
  defer dB_{[&] { cudaFree(dB); }};                                            \
                                                                               \
  state.add_global_memory_reads<DType>(static_cast<size_t>(N) * 2);            \
  state.add_global_memory_writes<DType>(static_cast<size_t>(N));               \
  state.add_element_count(ARITHMETIC_INTENSITY * 4, "FMA/row");                \
  state.add_element_count(N, "row_count");

// --- Benchmark normal FMA ---
void bench_fma_no_tma(nvbench::state &state) {
  BENCHMARK_PRELUDE

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &) {
    DType *dCx;
    cudaMalloc(&dCx, N * sizeof(DType));
    CHECK_CUDA_ERR();
    defer dCx_{[&] { cudaFree(dCx); }};
    cudaMemset(dCx, 0, N * sizeof(DType));
    CHECK_CUDA_ERR();

    i32 grid_size;
    i32 block_size;

    cudaOccupancyMaxPotentialBlockSizeWithFlags(
        &grid_size, &block_size, &fma_no_tma<ARITHMETIC_INTENSITY>, 0, 0, 0);
    CHECK_CUDA_ERR();
    fma_no_tma<ARITHMETIC_INTENSITY><<<grid_size, block_size>>>(dA, dB, dCx, N);
    CHECK_CUDA_ERR();
  });
}

// --- Benchmark normal FMA ---
void bench_fma_no_tma_no_restrict(nvbench::state &state) {
  BENCHMARK_PRELUDE

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &) {
    DType *dCx;
    cudaMalloc(&dCx, N * sizeof(DType));
    CHECK_CUDA_ERR();
    defer dCx_{[&] { cudaFree(dCx); }};
    cudaMemset(dCx, 0, N * sizeof(DType));
    CHECK_CUDA_ERR();

    i32 grid_size;
    i32 block_size;

    cudaOccupancyMaxPotentialBlockSizeWithFlags(
        &grid_size, &block_size, &fma_no_tma_no_restrict<ARITHMETIC_INTENSITY>,
        0, 0, 0);
    CHECK_CUDA_ERR();
    fma_no_tma_no_restrict<ARITHMETIC_INTENSITY>
        <<<grid_size, block_size>>>(dA, dB, dCx, N);
    CHECK_CUDA_ERR();
  });
}

// --- Benchmark FMA with static TILE_DIM and grid-strided ---
void bench_fma_tma_grid_strided(nvbench::state &state) {
  BENCHMARK_PRELUDE

  i32 grid_size;
  i32 block_size;

  cudaOccupancyMaxPotentialBlockSizeWithFlags(
      &grid_size, &block_size,
      &fma_tma_grid_strided<TILE_DIM, ARITHMETIC_INTENSITY>, 0, 0, 0);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &) {
    DType *dCx;
    cudaMalloc(&dCx, N * sizeof(DType));
    CHECK_CUDA_ERR();
    defer dCx_{[&] { cudaFree(dCx); }};
    cudaMemset(dCx, 0, N * sizeof(DType));
    CHECK_CUDA_ERR();

    CHECK_CUDA_ERR();
    fma_tma_grid_strided<TILE_DIM, ARITHMETIC_INTENSITY>
        <<<grid_size, block_size>>>(dA, dB, dCx, N);
    CHECK_CUDA_ERR();
  });
}

// --- Benchmark FMA with dynamic tuned shmem TMA ---
void bench_fma_tma_tuned(nvbench::state &state) {
  BENCHMARK_PRELUDE

  auto stats = tune_tma_kernel(&fma_tma_tuned_dyn_shmem<ARITHMETIC_INTENSITY>);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &) {
    DType *dCx;
    cudaMalloc(&dCx, N * sizeof(DType));
    CHECK_CUDA_ERR();
    defer dCx_{[&] { cudaFree(dCx); }};
    cudaMemset(dCx, 0, N * sizeof(DType));
    CHECK_CUDA_ERR();

    auto block_dim = tma_policy::block_dim;
    auto grid_dim = (N + stats.tile_dim - 1) / stats.tile_dim;
    auto bar_offset = 0;
    auto tile1_offset = alignup(bar_offset + sizeof(u64), BULK_COPY_ALIGNMENT);
    auto tile2_offset = alignup(tile1_offset + stats.tile_dim * sizeof(DType),
                                BULK_COPY_ALIGNMENT);

    // TODO: re-check all
    fma_tma_tuned_dyn_shmem<ARITHMETIC_INTENSITY>
        <<<grid_dim, block_dim, stats.smem_req>>>(dA, dB, bar_offset,
                                                  tile1_offset, tile2_offset,
                                                  stats.tile_dim, dCx, N);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERR();
  });
}

NVBENCH_BENCH(bench_fma_no_tma)
    .set_name("bench_fma_no_tma")
    .add_int64_power_of_two_axis("N", {30, 31, 32});

NVBENCH_BENCH(bench_fma_no_tma_no_restrict)
    .set_name("bench_fma_no_tma_no_restrict")
    .add_int64_power_of_two_axis("N", {30, 31, 32});

NVBENCH_BENCH(bench_fma_tma_grid_strided)
    .set_name("bench_fma_tma_grid_strided")
    .add_int64_power_of_two_axis("N", {30, 31, 32});

NVBENCH_BENCH(bench_fma_tma_tuned)
    .set_name("bench_fma_tma_tuned")
    .add_int64_power_of_two_axis("N", {30, 31, 32});

NVBENCH_MAIN
