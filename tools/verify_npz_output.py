#!/usr/bin/env python3
"""
验证 NPZ 输出与 CSR 文件的一致性

ABACUS NPZ 格式：
- lattice_vectors: 晶格向量 [9]
- atom_info: 原子信息 [nat, 5] (type, Z, x, y, z)
- orbital_info_X: 轨道信息 [nw, 3] (n, l, m)
- mat_iat1_iat2_Rx_Ry_Rz: 子矩阵块 [nw1, nw2]

使用方法：
    # 查看 NPZ 结构
    python verify_npz_output.py output_SR.npz
    
    # 转换为 CSR（只包含非空 R 向量）
    python verify_npz_output.py output_SR.npz --convert output_SR.csr
    
    # 转换为 CSR（使用参考文件的完整 R 向量列表，包括空的）
    python verify_npz_output.py output_SR.npz --convert output_SR.csr reference.csr
    
    # 对比统计信息
    python verify_npz_output.py output_SR.npz --compare original.csr
    
    # 深度比较两个 CSR 文件（跳过空 R 向量，只比较有数据的部分）
    python verify_npz_output.py output_SR.csr --diff original.csr
"""

import numpy as np
import os
import sys
from collections import defaultdict


def print_npz_structure(npz_file):
    """打印 NPZ 文件结构"""
    print(f"\n{'='*60}")
    print(f"NPZ 文件结构: {npz_file}")
    print('='*60)
    
    data = np.load(npz_file)
    print(f"包含的数组: {len(data.files)} 个")
    
    # 分类显示
    meta_keys = []
    mat_keys = []
    
    for key in data.files:
        if key.startswith('mat_'):
            mat_keys.append(key)
        else:
            meta_keys.append(key)
    
    print(f"\n元数据数组 ({len(meta_keys)}):")
    for key in meta_keys:
        arr = data[key]
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
    
    print(f"\n矩阵数据数组 ({len(mat_keys)}):")
    if len(mat_keys) > 5:
        for key in mat_keys[:3]:
            arr = data[key]
            print(f"  {key}: shape={arr.shape}")
        print(f"  ... ({len(mat_keys) - 5} more)")
        for key in mat_keys[-2:]:
            arr = data[key]
            print(f"  {key}: shape={arr.shape}")
    else:
        for key in mat_keys:
            arr = data[key]
            print(f"  {key}: shape={arr.shape}")
    
    # 统计 R 向量
    R_vectors = set()
    atom_pairs = set()
    total_elements = 0
    total_nonzero = 0
    
    for key in mat_keys:
        parts = key.split('_')
        if len(parts) >= 6:
            iat1, iat2 = int(parts[1]), int(parts[2])
            Rx, Ry, Rz = int(parts[3]), int(parts[4]), int(parts[5])
            R_vectors.add((Rx, Ry, Rz))
            atom_pairs.add((iat1, iat2))
            
            arr = data[key]
            total_elements += arr.size
            total_nonzero += np.count_nonzero(np.abs(arr) > 1e-10)
    
    print(f"\n统计:")
    print(f"  R 向量数: {len(R_vectors)}")
    print(f"  原子对数: {len(atom_pairs)}")
    print(f"  总元素数: {total_elements:,}")
    print(f"  非零元素 (>1e-10): {total_nonzero:,}")
    print(f"  文件大小: {os.path.getsize(npz_file) / 1024 / 1024:.2f} MB")


def get_orbital_info(data):
    """从 NPZ 数据中提取轨道信息"""
    # 获取原子信息
    atom_info = data['atom_info']
    nat = atom_info.shape[0]
    
    # 获取每个原子类型的轨道数
    ntype = 0
    for i in range(nat):
        it = int(atom_info[i, 0])
        ntype = max(ntype, it + 1)
    
    # 获取每个类型的轨道数
    nw_per_type = {}
    for it in range(ntype):
        key = f'orbital_info_{it}'
        if key in data.files:
            nw_per_type[it] = data[key].shape[0]
        else:
            nw_per_type[it] = 0
    
    # 计算每个原子的起始轨道索引
    atom_start_orb = []
    current_orb = 0
    for iat in range(nat):
        atom_start_orb.append(current_orb)
        it = int(atom_info[iat, 0])
        current_orb += nw_per_type[it]
    
    nlocal = current_orb
    
    return nat, nlocal, atom_info, nw_per_type, atom_start_orb


def npz_to_csr_format(npz_file, output_file, matrix_type='S', reference_csr=None):
    """
    将 NPZ 文件转换为 ABACUS CSR 格式
    
    注意：NPZ 只存储 atom_i <= atom_j 的原子对，需要利用对称性恢复：
    H_{mu,nu,R} = H_{nu,mu,-R}
    
    如果提供 reference_csr，则使用相同的 R 向量列表（包括空的 R 向量）
    """
    print(f"\n{'='*60}")
    print(f"将 NPZ 转换为 CSR 格式")
    print('='*60)
    
    data = np.load(npz_file)
    
    # 获取轨道信息
    nat, nlocal, atom_info, nw_per_type, atom_start_orb = get_orbital_info(data)
    print(f"  原子数: {nat}")
    print(f"  轨道数 (nlocal): {nlocal}")
    
    # 收集所有矩阵数据，按 R 向量分组
    # R_data[R] = {(global_row, global_col): value}
    R_data = defaultdict(dict)
    
    mat_count = 0
    for key in data.files:
        if not key.startswith('mat_'):
            continue
        
        parts = key.split('_')
        if len(parts) < 6:
            continue
        
        iat1 = int(parts[1])
        iat2 = int(parts[2])
        Rx = int(parts[3])
        Ry = int(parts[4])
        Rz = int(parts[5])
        
        R_key = (Rx, Ry, Rz)
        submat = data[key]
        
        # 计算全局轨道索引
        it1 = int(atom_info[iat1, 0])
        it2 = int(atom_info[iat2, 0])
        nw1 = nw_per_type[it1]
        nw2 = nw_per_type[it2]
        start1 = atom_start_orb[iat1]
        start2 = atom_start_orb[iat2]
        
        # 将子矩阵转换为全局索引
        for i in range(submat.shape[0]):
            for j in range(submat.shape[1]):
                val = submat[i, j]
                if abs(val) > 1e-10:
                    global_row = start1 + i
                    global_col = start2 + j
                    R_data[R_key][(global_row, global_col)] = val
        
        # 利用对称性：H_{mu,nu,R} = H_{nu,mu,-R}
        # 只对 R != 0 或 iat1 != iat2 的情况添加对称项
        if Rx != 0 or Ry != 0 or Rz != 0:
            R_key_neg = (-Rx, -Ry, -Rz)
            for i in range(submat.shape[0]):
                for j in range(submat.shape[1]):
                    val = submat[i, j]
                    if abs(val) > 1e-10:
                        # 对称：(iat2, iat1, -R) 的 (j, i) 元素
                        global_row = start2 + j
                        global_col = start1 + i
                        R_data[R_key_neg][(global_row, global_col)] = val
        elif iat1 != iat2:
            # R = 0 但 iat1 != iat2，添加 (iat2, iat1, 0) 的对称项
            for i in range(submat.shape[0]):
                for j in range(submat.shape[1]):
                    val = submat[i, j]
                    if abs(val) > 1e-10:
                        global_row = start2 + j
                        global_col = start1 + i
                        R_data[R_key][(global_row, global_col)] = val
        
        mat_count += 1
    
    print(f"  处理了 {mat_count} 个子矩阵块")
    print(f"  R 向量数（含对称）: {len(R_data)}")
    
    # 写入 CSR 格式
    type_str = 'S(R)' if matrix_type == 'S' else 'H(R)'
    
    # 获取 R 向量列表
    if reference_csr and os.path.exists(reference_csr):
        # 从参考文件读取完整的 R 向量列表（包括空的）
        sorted_R = []
        with open(reference_csr, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    try:
                        rx, ry, rz = int(parts[0]), int(parts[1]), int(parts[2])
                        sorted_R.append((rx, ry, rz))
                    except ValueError:
                        pass
        print(f"  从参考文件读取 {len(sorted_R)} 个 R 向量")
    else:
        # 按 R 向量排序（与 ABACUS 相同的顺序）
        sorted_R = sorted(R_data.keys())
    
    with open(output_file, 'w') as f:
        f.write(f"STEP: 0\n")
        f.write(f"Matrix Dimension of {type_str}: {nlocal}\n")
        f.write(f"Matrix number of {type_str}: {len(sorted_R)}\n")
        
        for R_key in sorted_R:
            Rx, Ry, Rz = R_key
            elements = R_data[R_key]
            
            # 按行列排序
            sorted_elements = sorted(elements.items(), key=lambda x: (x[0][0], x[0][1]))
            nnz = len(sorted_elements)
            
            f.write(f"{Rx} {Ry} {Rz} {nnz}\n")
            
            if nnz > 0:
                # 构建 CSR 格式
                values = []
                col_indices = []
                row_ptr = [0]
                
                current_row = 0
                for (row, col), val in sorted_elements:
                    while current_row < row:
                        row_ptr.append(len(values))
                        current_row += 1
                    values.append(val)
                    col_indices.append(col)
                
                # 填充剩余的 row_ptr
                while len(row_ptr) <= nlocal:
                    row_ptr.append(len(values))
                
                # 写入 values
                value_strs = [f"{v:.8e}" for v in values]
                f.write(" " + " ".join(value_strs) + "\n")
                
                # 写入 col_indices
                f.write(" " + " ".join(str(c) for c in col_indices) + "\n")
                
                # 写入 row_ptr
                f.write(" " + " ".join(str(r) for r in row_ptr) + "\n")
    
    print(f"  写入: {output_file}")
    print(f"  文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")


def compare_with_original(npz_file, original_csr_file):
    """对比 NPZ 和原始 CSR 文件的统计信息"""
    print(f"\n{'='*60}")
    print(f"对比统计信息")
    print('='*60)
    
    # 读取 NPZ
    data = np.load(npz_file)
    
    npz_R_vectors = set()
    npz_nonzero = 0
    
    for key in data.files:
        if not key.startswith('mat_'):
            continue
        parts = key.split('_')
        if len(parts) >= 6:
            Rx, Ry, Rz = int(parts[3]), int(parts[4]), int(parts[5])
            npz_R_vectors.add((Rx, Ry, Rz))
            arr = data[key]
            npz_nonzero += np.count_nonzero(np.abs(arr) > 1e-10)
    
    print(f"NPZ 文件: {npz_file}")
    print(f"  R 向量数: {len(npz_R_vectors)}")
    print(f"  非零元素: {npz_nonzero:,}")
    
    # 读取原始 CSR
    if os.path.exists(original_csr_file):
        csr_total_nnz = 0
        csr_num_R = 0
        csr_R_vectors = set()
        csr_nonzero_R = 0  # 非空 R 向量数
        
        with open(original_csr_file, 'r') as f:
            for line in f:
                if line.startswith('Matrix number'):
                    csr_num_R = int(line.split(':')[1].strip())
                parts = line.strip().split()
                if len(parts) == 4:
                    try:
                        rx, ry, rz, nnz = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                        csr_R_vectors.add((rx, ry, rz))
                        csr_total_nnz += nnz
                        if nnz > 0:
                            csr_nonzero_R += 1
                    except ValueError:
                        pass
        
        print(f"\n原始 CSR: {original_csr_file}")
        print(f"  R 向量数: {csr_num_R} (非空: {csr_nonzero_R}, 空: {csr_num_R - csr_nonzero_R})")
        print(f"  总非零元素: {csr_total_nnz:,}")
        
        # 对比
        print(f"\n对比结果:")
        if len(npz_R_vectors) == csr_nonzero_R:
            print(f"  ✅ 非空 R 向量数一致: {len(npz_R_vectors)}")
        else:
            print(f"  ⚠️  R 向量数: NPZ={len(npz_R_vectors)}, CSR非空={csr_nonzero_R}")
        
        if npz_nonzero == csr_total_nnz:
            print(f"  ✅ 非零元素数一致: {npz_nonzero:,}")
        else:
            print(f"  ❌ 非零元素数不一致: NPZ={npz_nonzero:,}, CSR={csr_total_nnz:,}")
    else:
        print(f"\n原始 CSR 文件不存在: {original_csr_file}")


def compare_csr_files(csr_file1, csr_file2):
    """
    深度比较两个 CSR 文件，跳过空的 R 向量（nnz=0）
    只比较有数据的 R 向量的实际内容
    """
    print(f"\n{'='*60}")
    print(f"深度比较两个 CSR 文件（跳过空 R 向量）")
    print('='*60)
    print(f"  文件1: {csr_file1}")
    print(f"  文件2: {csr_file2}")
    
    def parse_csr_file(filepath):
        """解析 CSR 文件，返回 {R: {'nnz': n, 'values': [...], 'cols': [...], 'row_ptr': [...]}}"""
        data = {}
        dimension = 0
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('Matrix Dimension'):
                dimension = int(line.split(':')[1].strip())
                i += 1
                continue
            
            if line.startswith('STEP') or line.startswith('Matrix number'):
                i += 1
                continue
            
            parts = line.split()
            if len(parts) == 4:
                try:
                    rx, ry, rz, nnz = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    R_key = (rx, ry, rz)
                    
                    if nnz == 0:
                        # 空 R 向量，跳过
                        data[R_key] = {'nnz': 0, 'values': [], 'cols': [], 'row_ptr': []}
                        i += 1
                    else:
                        # 读取 values, cols, row_ptr
                        i += 1
                        values_line = lines[i].strip() if i < len(lines) else ""
                        i += 1
                        cols_line = lines[i].strip() if i < len(lines) else ""
                        i += 1
                        row_ptr_line = lines[i].strip() if i < len(lines) else ""
                        i += 1
                        
                        # 解析 values
                        values = []
                        for v in values_line.split():
                            try:
                                values.append(float(v))
                            except:
                                pass
                        
                        # 解析 cols
                        cols = []
                        for c in cols_line.split():
                            try:
                                cols.append(int(c))
                            except:
                                pass
                        
                        # 解析 row_ptr
                        row_ptr = []
                        for r in row_ptr_line.split():
                            try:
                                row_ptr.append(int(r))
                            except:
                                pass
                        
                        data[R_key] = {'nnz': nnz, 'values': values, 'cols': cols, 'row_ptr': row_ptr}
                except ValueError:
                    i += 1
            else:
                i += 1
        
        return data, dimension
    
    # 解析两个文件
    print("\n解析文件...")
    data1, dim1 = parse_csr_file(csr_file1)
    data2, dim2 = parse_csr_file(csr_file2)
    
    print(f"  文件1: {len(data1)} 个 R 向量, 维度 {dim1}")
    print(f"  文件2: {len(data2)} 个 R 向量, 维度 {dim2}")
    
    # 获取非空 R 向量
    nonzero_R1 = {R for R, d in data1.items() if d['nnz'] > 0}
    nonzero_R2 = {R for R, d in data2.items() if d['nnz'] > 0}
    
    print(f"  文件1 非空 R: {len(nonzero_R1)}")
    print(f"  文件2 非空 R: {len(nonzero_R2)}")
    
    # 比较非空 R 向量集合
    only_in_1 = nonzero_R1 - nonzero_R2
    only_in_2 = nonzero_R2 - nonzero_R1
    common_R = nonzero_R1 & nonzero_R2
    
    print(f"\n比较非空 R 向量:")
    if only_in_1:
        print(f"  ⚠️  仅在文件1中: {len(only_in_1)} 个")
        for R in list(only_in_1)[:5]:
            print(f"      {R}")
    if only_in_2:
        print(f"  ⚠️  仅在文件2中: {len(only_in_2)} 个")
        for R in list(only_in_2)[:5]:
            print(f"      {R}")
    
    if not only_in_1 and not only_in_2:
        print(f"  ✅ 非空 R 向量集合完全一致: {len(common_R)} 个")
    
    # 比较每个共同 R 向量的数据
    print(f"\n比较共同 R 向量的数据内容...")
    differences = []
    total_nnz1 = 0
    total_nnz2 = 0
    
    for R in common_R:
        d1 = data1[R]
        d2 = data2[R]
        total_nnz1 += d1['nnz']
        total_nnz2 += d2['nnz']
        
        if d1['nnz'] != d2['nnz']:
            differences.append((R, f"nnz不同: {d1['nnz']} vs {d2['nnz']}"))
            continue
        
        if d1['cols'] != d2['cols']:
            differences.append((R, f"列索引不同"))
            continue
        
        if d1['row_ptr'] != d2['row_ptr']:
            differences.append((R, f"行指针不同"))
            continue
        
        # 比较值（允许小误差）
        if len(d1['values']) != len(d2['values']):
            differences.append((R, f"值数量不同: {len(d1['values'])} vs {len(d2['values'])}"))
            continue
        
        max_diff = 0
        for v1, v2 in zip(d1['values'], d2['values']):
            diff = abs(v1 - v2)
            max_diff = max(max_diff, diff)
        
        if max_diff > 1e-6:
            differences.append((R, f"值差异过大: max_diff={max_diff:.2e}"))
    
    print(f"  总非零元素: 文件1={total_nnz1:,}, 文件2={total_nnz2:,}")
    
    if differences:
        print(f"\n❌ 发现 {len(differences)} 个 R 向量有差异:")
        for R, msg in differences[:10]:
            print(f"    R={R}: {msg}")
        if len(differences) > 10:
            print(f"    ... 还有 {len(differences) - 10} 个差异")
    else:
        print(f"\n✅ 所有 {len(common_R)} 个共同 R 向量的数据完全一致！")
        print(f"   （跳过了文件2中的 {len(data2) - len(nonzero_R2)} 个空 R 向量）")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    npz_file = sys.argv[1]
    
    if not os.path.exists(npz_file):
        print(f"Error: 文件不存在: {npz_file}")
        sys.exit(1)
    
    if len(sys.argv) >= 3 and sys.argv[2] == '--convert':
        # 转换模式
        output_file = sys.argv[3] if len(sys.argv) > 3 else 'output.csr'
        reference_csr = sys.argv[4] if len(sys.argv) > 4 else None
        matrix_type = 'S' if 'SR' in npz_file or 'output_SR' in npz_file else 'H'
        npz_to_csr_format(npz_file, output_file, matrix_type, reference_csr)
        
    elif len(sys.argv) >= 3 and sys.argv[2] == '--compare':
        # 对比模式
        original_file = sys.argv[3] if len(sys.argv) > 3 else None
        if original_file:
            compare_with_original(npz_file, original_file)
        else:
            print("Usage: script.py npz_file --compare original_csr_file")
    
    elif len(sys.argv) >= 4 and sys.argv[2] == '--diff':
        # 深度比较两个 CSR 文件（跳过空 R 向量）
        csr_file2 = sys.argv[3]
        compare_csr_files(npz_file, csr_file2)  # npz_file 这里实际上是 csr_file1
    else:
        # 只打印结构
        print_npz_structure(npz_file)


if __name__ == '__main__':
    main()
