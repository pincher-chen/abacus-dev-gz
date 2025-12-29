# ABACUS H(R)/S(R) NPZ 压缩输出功能

## 1. 功能概述

添加 `out_hsr_npz` 参数，同时输出 H(R) 和 S(R) 矩阵的 NPZ 压缩格式，实现最优压缩率。

## 2. 修改的源码文件

| 文件 | 修改行数 | 说明 |
|------|----------|------|
| `source/module_parameter/input_parameter.h` | +1 行 | 添加 `bool out_hsr_npz = false;` |
| `source/module_io/read_input_item_output.cpp` | +15 行 | 添加参数读取和 CNPY 依赖检查 |
| `source/module_esolver/esolver_ks_lcao.cpp` | +20 行 | 添加 H(R) + S(R) NPZ 输出代码 |

**总计：约 36 行代码修改**

## 3. 实现逻辑

### 3.1 参数定义

```cpp
// source/module_parameter/input_parameter.h
bool out_hsr_npz = false;  // 同时输出 H(R) 和 S(R) 的 NPZ 格式
```

### 3.2 参数读取

```cpp
// source/module_io/read_input_item_output.cpp
Input_Item item("out_hsr_npz");
item.annotation = "output both H(R) and S(R) matrices in npz format";
read_sync_bool(input.out_hsr_npz);
// 检查 CNPY 库依赖
```

### 3.3 输出实现

```cpp
// source/module_esolver/esolver_ks_lcao.cpp
if (PARAM.inp.out_hsr_npz) {
    hamilt::HamiltLCAO<...>* p_ham_lcao = ...;
    
    // 1. 输出 S(R) 矩阵
    ModuleIO::output_mat_npz(ucell, "output_SR.npz", *(p_ham_lcao->getSR()));
    
    // 2. 输出 H(R) 矩阵 (spin 0)
    ModuleIO::output_mat_npz(ucell, "output_HR0.npz", *(p_ham_lcao->getHR()));
    
    // 3. 如果 nspin=2，输出 H(R) 矩阵 (spin 1)
    if (nspin == 2) {
        p_hamilt->updateHk(nks/2);  // 切换到 spin 1
        ModuleIO::output_mat_npz(ucell, "output_HR1.npz", *(p_ham_lcao->getHR()));
    }
}
```

## 4. 使用方法

在 INPUT 文件中添加：

```
out_hsr_npz    1
```

### 4.1 输出文件

| 文件 | 内容 |
|------|------|
| `output_SR.npz` | S(R) 重叠矩阵 |
| `output_HR0.npz` | H(R) 哈密顿矩阵 (spin 0) |
| `output_HR1.npz` | H(R) 哈密顿矩阵 (spin 1，仅 nspin=2) |

## 5. 压缩效率对比

### 5.1 测试数据

- 系统：174 个轨道，685 个 R 向量，nspin=2
- sparse_threshold: 1e-10

### 5.2 实测结果

| 格式 | S 矩阵 | H spin0 | H spin1 | 总大小 | 压缩率 |
|------|--------|---------|---------|--------|--------|
| **CSR 纯文本** | 81 MB | 138 MB | 138 MB | **357 MB** | 100% (基准) |
| **CSR 二进制** | 52 MB | 89 MB | 89 MB | **230 MB** | 64% |
| **NPZ (H+S)** | 22 MB | 41 MB | 41 MB | **104 MB** | **29%** ✅ 最优 |
| **HDF5** | 包含 | 包含 | 包含 | **154 MB** | 43% |

### 5.3 结论

- **NPZ 格式压缩率最优**：相比纯文本节省 **71%**（357MB → 104MB）
- 相比 HDF5 节省 **32%**（154MB → 104MB）

## 6. 验证准确性

### 6.1 验证脚本

使用 `verify_npz_output.py` 将 NPZ 转换为 CSR 格式并与原始文件对比：

```bash
# 转换 NPZ -> CSR
python verify_npz_output.py output_SR.npz --convert output_SR.csr

# 对比结果
diff output_SR.csr ../Scf_test4/OUT.Scf/data-SR-sparse_SPIN0.csr
```

### 6.2 预期结果

```
$ python /XYFS01/nscc-gz_pinchen_1/sf_box/abacus-develop-LTSv3.10.1/tools/verify_npz_output.py output_SR.csr --diff ../Scf_test4/OUT.Scf/data-SR-sparse_SPIN0.csr

============================================================
深度比较两个 CSR 文件（跳过空 R 向量）
============================================================
  文件1: output_SR.csr
  文件2: ../Scf_test4/OUT.Scf/data-SR-sparse_SPIN0.csr

解析文件...
  文件1: 441 个 R 向量, 维度 174
  文件2: 685 个 R 向量, 维度 174
  文件1 非空 R: 441
  文件2 非空 R: 441

比较非空 R 向量:
  ✅ 非空 R 向量集合完全一致: 441 个

比较共同 R 向量的数据内容...
  总非零元素: 文件1=4,453,720, 文件2=4,453,720

✅ 所有 441 个共同 R 向量的数据完全一致！
   （跳过了文件2中的 244 个空 R 向量）

```

## 7. NPZ vs HDF5 选择建议

| 场景 | 推荐格式 | 原因 |
|------|----------|------|
| 追求最小文件 | NPZ (`out_hsr_npz 1`) | 压缩率 29%，最小 |
| 追求便捷管理 | HDF5 (`out_mat_hs2_hdf5 1`) | 单文件，元数据丰富 |
| Python 后处理 | NPZ | 直接 `np.load()` 读取 |
| C++/Fortran 后处理 | HDF5 | 跨语言支持更好 |

