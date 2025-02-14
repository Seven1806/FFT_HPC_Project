# 🚀 Fast Fourier Transform (FFT) Performance Analysis

This project analyzes the performance of different FFT implementations on **CPU (Sequential & OpenMP) and GPU (CUDA)**.

## 📁 **Project Structure**

hpc_project/ ├── repo/ (Git repository with source code & compiled executables) │ ├── build/ (Contains compiled executables) │ │ ├── fft_4speeds │ │ ├── fft_gpu │ │ ├── fft_parallel │ │ ├── fft_sequencial │ ├── fft_sequencial.c │ ├── fft_parallel.c │ ├── fft_gpu.cu │ ├── fft_4speeds_clean.cu │ ├── CMakeLists.txt │ ├── ... ├── docs/ │ ├── report.pdf (Project Report) │ ├── poster.pdf



---

## **🖥 Implementations**
- **Sequential FFT (CPU)**
- **Parallel FFT (CPU - OpenMP)**
- **FFT using CUDA (GPU)**
- **Multi-version Performance Comparison (`fft_4speeds`)**

---

## **⚙️ Hardware Setup**
- **CPU:** AMD EPYC 7453 (16 Cores)
- **GPU:** NVIDIA A100-SXM4-80GB
- **RAM:** 16 GB
- **Software:** Ubuntu 20.04, CUDA 11.4, GCC 10.3, OpenMP 4.5

---

## **📊 Results & Speedup**
| FFT Size  | Sequential (CPU) | Parallel OpenMP (CPU) | CUDA GPU |
|-----------|-----------------|----------------------|---------|
| **4096**  | 0.000688s        | 0.151379s           | 0.339362s |
| **65536** | 0.010447s        | 0.196264s           | 0.000996s |
| **1048576** | 0.229312s      | 0.382001s           | 0.022231s |
| **16777216** | 4.534188s     | 0.801931s           | 0.058288s |

🚀 **Speedup Insights:**
- **For small FFTs (4K, 64K):** OpenMP **outperforms** GPU due to CUDA overhead.
- **For large FFTs (1M, 16M):** **CUDA provides ~70x acceleration** over CPU.

---

## **🛠 How to Compile & Run**
### **🔹 Compile using CMake**
```bash
cd repo/
mkdir -p build && cd build
cmake ..
make -j$(nproc)



