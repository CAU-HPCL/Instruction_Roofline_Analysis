# # cmake_minimum_required(VERSION 3.18)
# # project(spmm_all LANGUAGES CXX CUDA)

# # # ── (선택) Kokkos nvcc_wrapper 사용: CXX 컴파일러로 설정
# # #     Kokkos CUDA 백엔드일 때 권장. 없으면 자동으로 건너뜀.
# # if(NOT DEFINED CMAKE_CXX_COMPILER AND DEFINED ENV{KOKKOS_PREFIX}
# #    AND EXISTS "$ENV{KOKKOS_PREFIX}/bin/nvcc_wrapper")
# #   set(CMAKE_CXX_COMPILER "$ENV{KOKKOS_PREFIX}/bin/nvcc_wrapper" CACHE FILEPATH "" FORCE)
# # endif()

# # # ── 공통 표준/옵션
# # set(CMAKE_CXX_STANDARD 17)
# # set(CMAKE_CXX_STANDARD_REQUIRED ON)
# # set(CMAKE_CXX_EXTENSIONS OFF)

# # # CUDA 쪽도 맞춰 주기
# # set(CMAKE_CUDA_STANDARD 17)
# # set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# # # 아키텍처는 환경에 맞게 조정(예: 86=3090, 89=4090). 필요시 -DCMAKE_CUDA_ARCHITECTURES=89 로 넘겨도 됨.
# # set(CMAKE_CUDA_ARCHITECTURES 89)

# # set(BSA_OBJS 
# #   "/root/BSA-SpMM_EURO-PAR-2024/bin/matrices.o")



# # find_package(Kokkos         REQUIRED)
# # find_package(KokkosKernels  REQUIRED)
# # find_package(Ginkgo         REQUIRED)
# # find_package(CUDAToolkit    REQUIRED)
# # find_package(OpenMP         REQUIRED)

# # # ── 소스 (main이 .cu)
# # add_executable(spmm_all
# #   src/main.cu
# #   "/root/BSA-SpMM_EURO-PAR-2024/src/matrices.cpp"
# #   # 다른 .cpp/.cu 있으면 여기에 추가
# # )

# # # ── 링크: 임포트 타겟 사용(대부분의 include 경로/정의가 자동 전파됨)
# # target_link_libraries(spmm_all
# #   PRIVATE
# #     Kokkos::kokkos
# #     Kokkos::kokkoskernels
# #     Ginkgo::ginkgo
# #     CUDA::cusparse
# #     CUDA::cudart
# #     OpenMP::OpenMP_CXX
# #     # 필요시: CUDA::cublas, CUDA::cuda_driver 등
# # )

# # # ── (선택) 수동 include path를 추가하고 싶다면 여기에
# # #     보통은 위 target_link_libraries 로 충분하지만,
# # #     각 라이브러리의 설치 prefix를 직접 include에 넣고 싶을 때 사용.
# # target_include_directories(spmm_all PRIVATE
# #   "$ENV{KOKKOS_PREFIX}/include"
# #   "$ENV{KOKKOSKERNELS_PREFIX}/include"
# #   "$ENV{GKO_PREFIX}/include"
# #   "/root/BSA-SpMM_EURO-PAR-2024/include"
# #   "/root/ICTC/include"
# #   # 여기에 임의의 경로를 더 넣어도 됨:
# #   # "/your/extra/include"
# # )

# # # ── (선택) 런타임 RPATH를 박아서 LD_LIBRARY_PATH 없이 실행
# # set_target_properties(spmm_all PROPERTIES
# #   BUILD_RPATH   "$ENV{KOKKOS_PREFIX}/lib;$ENV{KOKKOSKERNELS_PREFIX}/lib;$ENV{GKO_PREFIX}/lib"
# #   INSTALL_RPATH "$ENV{KOKKOS_PREFIX}/lib;$ENV{KOKKOSKERNELS_PREFIX}/lib;$ENV{GKO_PREFIX}/lib"
# # )


# # if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
# #   target_compile_options(spmm_all PRIVATE
# #     $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
# #   )
# # endif()


# # # 이걸로 빌드

# # # mkdir -p build && cd build
# # # cmake .. -DCMAKE_BUILD_TYPE=Release \
# # #   -DGinkgo_DIR="$GKO_PREFIX/lib/cmake/Ginkgo" \
# # #   -DKokkos_DIR="$KOKKOS_PREFIX/lib/cmake/Kokkos" \
# # #   -DKokkosKernels_DIR="$KOKKOSKERNELS_PREFIX/lib/cmake/KokkosKernels" \
# # #   -DCMAKE_CUDA_ARCHITECTURES=89         # 4090이면
# # # cmake --build . -j

# cmake_minimum_required(VERSION 3.18)
# project(spmm_all LANGUAGES CXX CUDA)

# # ── (선택) Kokkos nvcc_wrapper 사용
# if(NOT DEFINED CMAKE_CXX_COMPILER AND DEFINED ENV{KOKKOS_PREFIX}
#    AND EXISTS "$ENV{KOKKOS_PREFIX}/bin/nvcc_wrapper")
#   set(CMAKE_CXX_COMPILER "$ENV{KOKKOS_PREFIX}/bin/nvcc_wrapper" CACHE FILEPATH "" FORCE)
# endif()

# # ── 표준
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# # ── GPU 아키텍처 (4090=89, 3090=86 등)
# # 필요시 -DCMAKE_CUDA_ARCHITECTURES=89 로 외부에서 덮어쓰기 가능
# if(NOT CMAKE_CUDA_ARCHITECTURES)
#   set(CMAKE_CUDA_ARCHITECTURES 89)
# endif()

# # ── 외부 파일(주의: .o 를 그대로 링크하면 호환성 이슈 가능)
# # 가급적 .cpp 원본을 추가하세요.
# # set(BSA_OBJS "/root/BSA-SpMM_EURO-PAR-2024/bin/matrices.o")

# find_package(Kokkos         REQUIRED)
# find_package(KokkosKernels  REQUIRED)   # ← 패키지명은 이게 맞음
# find_package(Ginkgo         REQUIRED)
# find_package(CUDAToolkit    REQUIRED)
# find_package(OpenMP         REQUIRED)

# add_executable(spmm_all
#   src/main.cu
#   /root/BSA-SpMM_EURO-PAR-2024/src/matrices.cpp
#   # ${BSA_OBJS}  # <- 정말 필요할 때만 사용
# )

# # ── 링크: KokkosKernels의 정식 임포트 타깃명은 'KokkosKernels::kokkoskernels'
# target_link_libraries(spmm_all PRIVATE
#   Kokkos::kokkos
#   KokkosKernels::kokkoskernels
#   Ginkgo::ginkgo
#   CUDA::cusparse
#   CUDA::cudart
#   OpenMP::OpenMP_CXX
#   # 필요시: CUDA::cublas, CUDA::cuda_driver
# )

# # ── (선택) include path 수동 추가
# target_include_directories(spmm_all PRIVATE
#   "$ENV{KOKKOS_PREFIX}/include"
#   "$ENV{KOKKOSKERNELS_PREFIX}/include"
#   "$ENV{GKO_PREFIX}/include"
#   "/root/BSA-SpMM_EURO-PAR-2024/include"
#   "/root/ICTC/include"
# )

# # ── RPATH (LD_LIBRARY_PATH 없이 실행)
# set_target_properties(spmm_all PROPERTIES
#   BUILD_RPATH   "$ENV{KOKKOS_PREFIX}/lib;$ENV{KOKKOSKERNELS_PREFIX}/lib;$ENV{GKO_PREFIX}/lib"
#   INSTALL_RPATH "$ENV{KOKKOS_PREFIX}/lib;$ENV{KOKKOSKERNELS_PREFIX}/lib;$ENV{GKO_PREFIX}/lib"
# )

# # ── 최적화: CXX/ CUDA 모두 -O3 -DNDEBUG
# target_compile_options(spmm_all PRIVATE
#   $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG>
#   $<$<COMPILE_LANGUAGE:CUDA>:-O3 -DNDEBUG>
# )

# # (선택) 호스트 벡터화
# if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
#   target_compile_options(spmm_all PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
# endif()

# if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
#   target_compile_options(spmm_all PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
#   )
# endif()


# # ── OpenMP 플래그 중복 피하기
# # OpenMP::OpenMP_CXX가 알아서 호스트 컴파일러에 -fopenmp를 넣어줌.
# # 아래 줄은 보통 불필요하니 주석으로 유지:
# # if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
# #   target_compile_options(spmm_all PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
# # endif()


cmake_minimum_required(VERSION 3.18)
project(spmm_all LANGUAGES CXX CUDA)

# ── (선택) Kokkos nvcc_wrapper 사용
if(NOT DEFINED CMAKE_CXX_COMPILER AND DEFINED ENV{KOKKOS_PREFIX}
   AND EXISTS "$ENV{KOKKOS_PREFIX}/bin/nvcc_wrapper")
  set(CMAKE_CXX_COMPILER "$ENV{KOKKOS_PREFIX}/bin/nvcc_wrapper" CACHE FILEPATH "" FORCE)
endif()

# ── 표준
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── GPU 아키텍처 (4090=89, 3090=86 등)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

# find_package(Kokkos         REQUIRED)
# find_package(KokkosKernels  REQUIRED)
find_package(Ginkgo         REQUIRED)
find_package(CUDAToolkit    REQUIRED)
find_package(OpenMP         REQUIRED)
find_library(NVTOOLSEXT nvToolsExt
  PATHS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu
  REQUIRED)

add_executable(spmm_all
  src/cuSPARSE_main.cu
  src/external/BSA-SpMM/src/matrices.cpp
)

target_link_libraries(spmm_all PRIVATE
  # Kokkos::kokkos
  # KokkosKernels::kokkoskernels
  Ginkgo::ginkgo
  CUDA::cusparse
  CUDA::cudart
  OpenMP::OpenMP_CXX
  ${NVTOOLSEXT}          # NVTX 링크
)

# (선택) include path 수동 추가
target_include_directories(spmm_all PRIVATE
  "$ENV{GKO_PREFIX}/include"
  "include"
  "src/external/BSA-SpMM/include"
)

# ── RPATH (LD_LIBRARY_PATH 없이 실행)
set_target_properties(spmm_all PROPERTIES
  BUILD_RPATH   "$ENV{GKO_PREFIX}/lib"
  INSTALL_RPATH "$ENV{GKO_PREFIX}/lib"
)

# ── 최적화: CXX/ CUDA 모두 -O3 -DNDEBUG
target_compile_options(spmm_all PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -DNDEBUG>
)

# (선택) 호스트 벡터화
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(spmm_all PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif()

if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  target_compile_options(spmm_all PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
  )
endif()

# NVCC 권장 옵션 (템플릿/람다 호환성)
target_compile_options(spmm_all PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
)

