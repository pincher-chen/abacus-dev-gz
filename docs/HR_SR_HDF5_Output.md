# ABACUS H(R)/S(R) HDF5 压缩输出功能

## 1. 修改的源码文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `source/module_io/write_HS_R_hdf5.cpp` | ~500 行（新建） | HDF5 写入核心实现 |
| `source/module_io/write_HS_R_hdf5.h` | ~50 行（新建） | 头文件声明 |
| `source/module_esolver/esolver_ks_lcao.cpp` | ~20 行（修改） | 调用 HDF5 输出 |
| `source/module_io/CMakeLists.txt` | ~5 行（修改） | 添加编译配置 |
| `source/module_io/input_conv.cpp` | ~5 行（修改） | 添加输入参数 |

## 2. 实现逻辑

### 2.1 核心算法（伪代码）

```
function output_HSR_hdf5(HS_Arrays, pv, sparse_threshold, compression_level):
    
    # Step 1: 计算每个 R 向量的非零元素数量
    for each R_coor in all_R_coor:
        H_nonzero_num[R] = count_nonzero(HR_sparse[R])
        S_nonzero_num[R] = count_nonzero(SR_sparse[R])
    
    # Step 2: MPI 归约（多进程合并）
    MPI_Reduce(H_nonzero_num)
    MPI_Reduce(S_nonzero_num)
    
    # Step 3: 确定要输出的 R 向量（与原始 CSR 逻辑一致）
    for idx in range(total_R_num):
        if H_nonzero_num[idx] > 0 or S_nonzero_num[idx] > 0:
            should_output[idx] = true
    
    # Step 4: 写入 HDF5（带 gzip 压缩）
    for each R_coor where should_output[R] == true:
        # MPI 归约矩阵数据
        for row in range(nlocal):
            line = collect_row_data(XR, row, pv)
            MPI_Reduce(line)  # 收集所有进程的数据
            
            # 构建 CSR 格式
            for col where abs(line[col]) > threshold:
                values.append(line[col])
                col_indices.append(col)
            row_ptr.append(len(values))
        
        # 写入 HDF5 dataset（gzip level 9）
        H5Dwrite(values, col_indices, row_ptr)
```

### 2.2 HDF5 文件结构

```
data-HSR_step0.h5
├── /metadata
│   ├── @step = 0
│   ├── @nspin = 2
│   ├── @dimension = 174
│   └── @num_R_vectors = 685
├── /R_vectors
│   └── coordinates [685, 3]  # 所有 R 向量坐标
├── /overlap
│   ├── S_R_0/
│   │   ├── @nnz = 1234
│   │   ├── values [1234]
│   │   ├── col_indices [1234]
│   │   └── row_ptr [175]
│   └── S_R_1/ ...
└── /hamiltonian
    ├── H_R_spin0_R0/ ...
    └── H_R_spin1_R0/ ...
```

## 3. 测试验证

### 3.1 验证方法

使用 Python 脚本将 HDF5 转换回 CSR 格式，与原始输出对比：

```bash
# 转换 HDF5 -> CSR
python verify_hdf5_output.py OUT.Scf/data-HSR_step0.h5 --convert output_csr/

# 对比结果
diff output_csr/data-SR-sparse_SPIN0.csr original/data-SR-sparse_SPIN0.csr
diff output_csr/data-HR-sparse_SPIN0.csr original/data-HR-sparse_SPIN0.csr
diff output_csr/data-HR-sparse_SPIN1.csr original/data-HR-sparse_SPIN1.csr
```

### 3.2 验证结果

```
$ diff output_csr/data-SR-sparse_SPIN0.csr ../Scf_test4/OUT.Scf/data-SR-sparse_SPIN0.csr 
（无输出 = 完全一致）

$ diff output_csr/data-HR-sparse_SPIN0.csr ../Scf_test4/OUT.Scf/data-HR-sparse_SPIN0.csr 
（无输出 = 完全一致）

$ diff output_csr/data-HR-sparse_SPIN1.csr ../Scf_test4/OUT.Scf/data-HR-sparse_SPIN1.csr 
（无输出 = 完全一致）
```

✅ **转换后的 CSR 文件与原始 ABACUS 输出完全一致，验证 HDF5 功能正确。**

## 4. 压缩效率对比

### 4.1 测试数据

- 系统：174 个轨道，685 个 R 向量，nspin=2
- sparse_threshold: 1e-10

### 4.2 四种格式对比

| 格式 | 参数 | S 矩阵 | H spin0 | H spin1 | 总大小 | 压缩率 |
|------|------|--------|---------|---------|--------|--------|
| **CSR 纯文本** | `out_mat_hs2 1` | 81 MB | 138 MB | 138 MB | **357 MB** | 100% (基准) |
| **CSR 二进制** | `out_mat_hs2 1` (binary) | 52 MB | 89 MB | 89 MB | **230 MB** | 64% |
| **NPZ (现有)** | `out_hr_npz 1` | ❌ 不支持 | 41 MB | 41 MB | **82 MB*** | 23% (仅H) |
| **NPZ (理论)** | 如果支持 S | ~24 MB | 41 MB | 41 MB | **~106 MB** | **~30%** (理论最优) |
| **HDF5** | `out_mat_hs2_hdf5 1` | ✅ 包含 | ✅ 包含 | ✅ 包含 | **154 MB** | **43%** |

> *NPZ 格式目前不支持 S(R) 输出，82 MB 仅包含 H 矩阵  
> **理论 NPZ 值基于 S/H 比例估算：S_npz ≈ 41 MB × (81/138) ≈ 24 MB

### 4.3 分析

| 方法 | 优点 | 缺点 |
|------|------|------|
| **CSR 纯文本** | 可读性好，兼容性强 | 文件最大，无压缩 |
| **CSR 二进制** | 比文本节省 36% | 仍较大，多文件管理 |
| **NPZ (现有)** | 压缩率高(23%) | ❌ 不支持 S(R) 输出 |
| **NPZ (理论)** | 压缩率理论最优(~30%) | 需要扩展代码支持 S(R)，多文件管理 |
| **HDF5** | 支持 S+H 完整输出，单文件管理，元数据丰富 | 压缩率 43% (比理论 NPZ 大 45%) |

### 4.4 HDF5 vs NPZ 权衡

| 特性 | NPZ (理论完整版) | HDF5 (本实现) | 说明 |
|------|------------------|---------------|------|
| **压缩率** | ~30% ✅ 更优 | 43% | NPZ 更小 (~106 MB vs 154 MB) |
| **S(R) 支持** | 需要扩展代码 | ✅ 已支持 | HDF5 现成可用 |
| **文件数量** | 3 个文件 (S, H0, H1) | ✅ 1 个文件 | HDF5 便于管理 |
| **元数据** | 有限 | ✅ 丰富 | HDF5 可存储 step, nspin, R 坐标等 |
| **并行 I/O** | 不支持 | ✅ 支持 | HDF5 支持 MPI-IO |
| **跨语言** | Python 限定 | ✅ C++/Python/Fortran | HDF5 生态更广 |

### 4.5 结论

**为什么选择 HDF5？**

1. **即用性**：NPZ 目前不支持 S(R)，需要额外开发；HDF5 现成可用
2. **单文件**：154 MB 单文件 vs ~106 MB 三个文件，便于传输和归档
3. **元数据**：HDF5 内置存储 R 坐标、step、nspin 等，无需额外文件
4. **可扩展性**：HDF5 支持并行 I/O、增量写入、部分读取等高级特性
5. **存储效率**：相比纯文本节省 **57%**（357MB → 154MB），相比二进制节省 **33%**

**未来优化方向**：
- 可进一步降低 HDF5 压缩率，使用更激进的压缩算法（如 LZF、BLOSC）
- 或扩展 NPZ 支持 S(R)，实现理论最优压缩

## 5. 使用方法

在 INPUT 文件中添加：

```
out_mat_hs2_hdf5    1
```

输出文件：`OUT.xxx/data-HSR_step0.h5`

