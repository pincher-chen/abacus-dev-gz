#!/usr/bin/env python3
"""
验证 HDF5 输出与 CSR 文件的一致性

使用方法：
    python verify_hdf5_output.py <hdf5_file>
    python verify_hdf5_output.py <hdf5_file> <csr_directory>
    python verify_hdf5_output.py <hdf5_file> --convert <output_directory>

示例：
    # 查看 HDF5 文件结构
    python verify_hdf5_output.py OUT.Scf/data-HSR_step0.h5

    # 与原始 CSR 文件对比
    python verify_hdf5_output.py OUT.Scf/data-HSR_step0.h5 OUT.Scf_bak/

    # 转换为 ABACUS 原始 CSR 格式
    python verify_hdf5_output.py OUT.Scf/data-HSR_step0.h5 --convert output_csr/
"""

import h5py
import numpy as np
import os
import sys
from collections import defaultdict


def print_hdf5_structure(hdf5_file):
    """打印 HDF5 文件结构"""
    print(f"\n{'='*60}")
    print(f"HDF5 文件结构: {hdf5_file}")
    print('='*60)
    
    with h5py.File(hdf5_file, 'r') as f:
        def print_item(name, obj):
            indent = '  ' * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}[Dataset] {name}: shape={obj.shape}, dtype={obj.dtype}")
            else:
                print(f"{indent}[Group] {name}/")
                for attr_name, attr_value in obj.attrs.items():
                    print(f"{indent}  @{attr_name} = {attr_value}")
        
        f.visititems(print_item)
        
        print("\n元数据:")
        if '/metadata' in f:
            for attr_name, attr_value in f['/metadata'].attrs.items():
                print(f"  {attr_name} = {attr_value}")
        
        # 统计信息
        print("\n统计信息:")
        total_S_nnz = 0
        total_H_nnz = 0
        S_count = 0
        S_nonzero = 0
        H_count = 0
        H_nonzero = 0
        
        if '/overlap' in f:
            for key in f['/overlap'].keys():
                grp = f[f'/overlap/{key}']
                S_count += 1
                if 'nnz' in grp.attrs:
                    nnz = grp.attrs['nnz']
                    total_S_nnz += nnz
                    if nnz > 0:
                        S_nonzero += 1
        
        if '/hamiltonian' in f:
            for key in f['/hamiltonian'].keys():
                grp = f[f'/hamiltonian/{key}']
                H_count += 1
                if 'nnz' in grp.attrs:
                    nnz = grp.attrs['nnz']
                    total_H_nnz += nnz
                    if nnz > 0:
                        H_nonzero += 1
        
        print(f"  S 矩阵: {S_count} 个 R 向量 (非零: {S_nonzero}), 总非零元素 {total_S_nnz}")
        print(f"  H 矩阵: {H_count} 个 (spin, R) 组合 (非零: {H_nonzero}), 总非零元素 {total_H_nnz}")


def convert_hdf5_to_abacus_csr(hdf5_file, output_dir):
    """
    将 HDF5 文件转换为 ABACUS 原始 CSR 格式
    
    ABACUS CSR 格式：
    STEP: <step>
    Matrix Dimension of S(R)/H(R): <dim>
    Matrix number of S(R)/H(R): <num_R_vectors>
    <rx> <ry> <rz> <nnz>
    <values...> (一行，科学计数法，8位精度)
    <col_indices...> (一行)
    <row_ptr...> (一行)
    """
    print(f"\n{'='*60}")
    print(f"将 HDF5 转换为 ABACUS 原始 CSR 格式")
    print('='*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(hdf5_file, 'r') as f:
        # 获取元数据
        dim = f['/metadata'].attrs['dimension']
        nspin = f['/metadata'].attrs['nspin']
        step = f['/metadata'].attrs['step']
        num_R = f['/metadata'].attrs['num_R_vectors']
        
        # 读取 R 向量坐标
        R_coords = []
        if '/R_vectors/coordinates' in f:
            R_coords = f['/R_vectors/coordinates'][:]
        
        # 收集所有矩阵数据
        S_data = {}  # {R_idx: (rx, ry, rz, values, col_indices, row_ptr)}
        H_data = defaultdict(dict)  # {spin: {R_idx: (rx, ry, rz, values, col_indices, row_ptr)}}
        
        # 读取 S 矩阵
        if '/overlap' in f:
            for key in sorted(f['/overlap'].keys()):
                if not key.startswith('S_R_'):
                    continue
                R_idx = int(key.split('_')[-1])
                grp = f[f'/overlap/{key}']
                
                # 获取 R 向量坐标
                if R_idx < len(R_coords):
                    rx, ry, rz = R_coords[R_idx]
                else:
                    rx, ry, rz = 0, 0, 0
                
                # 检查是否为空矩阵
                if 'values' not in grp:
                    # 空矩阵：nnz=0
                    nnz = grp.attrs['nnz'] if 'nnz' in grp.attrs else 0
                    S_data[R_idx] = (int(rx), int(ry), int(rz), 
                                    np.array([]), np.array([]), np.zeros(dim + 1, dtype=int))
                else:
                    values = grp['values'][:]
                    col_indices = grp['col_indices'][:]
                    row_ptr = grp['row_ptr'][:]
                    S_data[R_idx] = (int(rx), int(ry), int(rz), values, col_indices, row_ptr)
        
        # 读取 H 矩阵
        if '/hamiltonian' in f:
            for key in sorted(f['/hamiltonian'].keys()):
                grp = f[f'/hamiltonian/{key}']
                
                # 解析 spin 和 R index
                parts = key.split('_')
                spin = 0
                R_idx = 0
                for p in parts:
                    if p.startswith('spin'):
                        spin = int(p[4:])
                    if p.startswith('R') and p[1:].isdigit():
                        R_idx = int(p[1:])
                
                if R_idx < len(R_coords):
                    rx, ry, rz = R_coords[R_idx]
                else:
                    rx, ry, rz = 0, 0, 0
                
                # 检查是否为空矩阵
                if 'values' not in grp:
                    nnz = grp.attrs['nnz'] if 'nnz' in grp.attrs else 0
                    H_data[spin][R_idx] = (int(rx), int(ry), int(rz),
                                          np.array([]), np.array([]), np.zeros(dim + 1, dtype=int))
                else:
                    values = grp['values'][:]
                    col_indices = grp['col_indices'][:]
                    row_ptr = grp['row_ptr'][:]
                    H_data[spin][R_idx] = (int(rx), int(ry), int(rz), values, col_indices, row_ptr)
        
        # 写入 S 矩阵 CSR 文件
        S_output = os.path.join(output_dir, 'data-SR-sparse_SPIN0.csr')
        print(f"写入: {S_output}")
        write_abacus_csr(S_output, step, dim, S_data, 'S')
        
        # 写入 H 矩阵 CSR 文件
        for spin in sorted(H_data.keys()):
            H_output = os.path.join(output_dir, f'data-HR-sparse_SPIN{spin}.csr')
            print(f"写入: {H_output}")
            write_abacus_csr(H_output, step, dim, H_data[spin], 'H')
    
    print(f"\n转换完成！输出目录: {output_dir}")


def write_abacus_csr(filename, step, dim, data_dict, matrix_type):
    """
    写入 ABACUS 原始 CSR 格式文件
    
    格式：
    STEP: <step>
    Matrix Dimension of S(R)/H(R): <dim>
    Matrix number of S(R)/H(R): <num_R_vectors>
    <rx> <ry> <rz> <nnz>
    <values...>
    <col_indices...>
    <row_ptr...>
    """
    num_R = len(data_dict)
    type_str = 'S(R)' if matrix_type == 'S' else 'H(R)'
    
    with open(filename, 'w') as f:
        # 写入头部
        f.write(f"STEP: {step}\n")
        f.write(f"Matrix Dimension of {type_str}: {dim}\n")
        f.write(f"Matrix number of {type_str}: {num_R}\n")
        
        # 按 R_idx 顺序写入每个 R 向量的数据
        for R_idx in sorted(data_dict.keys()):
            rx, ry, rz, values, col_indices, row_ptr = data_dict[R_idx]
            nnz = len(values)
            
            # 写入 R 向量和 nnz
            f.write(f"{rx} {ry} {rz} {nnz}\n")
            
            if nnz > 0:
                # 写入 values（科学计数法，8位精度）
                value_strs = []
                for v in values:
                    if isinstance(v, (complex, np.complexfloating)):
                        # 复数格式
                        value_strs.append(f"({v.real:.8e},{v.imag:.8e})")
                    elif isinstance(v, np.ndarray) and len(v) == 2:
                        # HDF5 复数存储格式
                        value_strs.append(f"({v[0]:.8e},{v[1]:.8e})")
                    else:
                        value_strs.append(f"{float(v):.8e}")
                f.write(" " + " ".join(value_strs) + "\n")
                
                # 写入 col_indices
                f.write(" " + " ".join(str(int(c)) for c in col_indices) + "\n")
                
                # 写入 row_ptr
                f.write(" " + " ".join(str(int(r)) for r in row_ptr) + "\n")


def compare_csr_files(hdf5_converted_dir, original_dir):
    """对比转换后的 CSR 文件与原始文件"""
    print(f"\n{'='*60}")
    print("对比 CSR 文件")
    print('='*60)
    
    files_to_compare = [
        'data-SR-sparse_SPIN0.csr',
        'data-HR-sparse_SPIN0.csr',
        'data-HR-sparse_SPIN1.csr'
    ]
    
    for filename in files_to_compare:
        converted_file = os.path.join(hdf5_converted_dir, filename)
        original_file = os.path.join(original_dir, filename)
        
        if not os.path.exists(converted_file):
            print(f"跳过: {filename} (转换文件不存在)")
            continue
        
        if not os.path.exists(original_file):
            print(f"跳过: {filename} (原始文件不存在)")
            continue
        
        # 读取并对比
        converted_info = analyze_csr_file(converted_file)
        original_info = analyze_csr_file(original_file)
        
        converted_size = os.path.getsize(converted_file)
        original_size = os.path.getsize(original_file)
        
        nnz_match = converted_info['total_nnz'] == original_info['total_nnz']
        R_match = converted_info['num_R'] == original_info['num_R']
        
        status = "✓ 完全一致" if (nnz_match and R_match) else "✗ 不一致"
        
        print(f"\n{filename}:")
        print(f"  原始文件:")
        print(f"    R 向量数: {original_info['num_R']}, 非零 R: {original_info['nonzero_R']}, 总 nnz: {original_info['total_nnz']:,}, 大小: {original_size/1024/1024:.1f} MB")
        print(f"  转换文件:")
        print(f"    R 向量数: {converted_info['num_R']}, 非零 R: {converted_info['nonzero_R']}, 总 nnz: {converted_info['total_nnz']:,}, 大小: {converted_size/1024/1024:.1f} MB")
        print(f"  状态: {status}")


def analyze_csr_file(filename):
    """分析 CSR 文件的统计信息"""
    total_nnz = 0
    num_R = 0
    nonzero_R = 0
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Matrix number'):
                num_R = int(line.split(':')[1].strip())
                continue
            
            parts = line.split()
            # 查找 "rx ry rz nnz" 格式的行
            if len(parts) == 4:
                try:
                    rx, ry, rz, nnz = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    total_nnz += nnz
                    if nnz > 0:
                        nonzero_R += 1
                except ValueError:
                    pass
    
    return {
        'num_R': num_R,
        'nonzero_R': nonzero_R,
        'total_nnz': total_nnz
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    hdf5_file = sys.argv[1]
    
    if not os.path.exists(hdf5_file):
        print(f"Error: 文件不存在: {hdf5_file}")
        sys.exit(1)
    
    if len(sys.argv) >= 3:
        if sys.argv[2] == '--convert':
            # 转换模式
            output_dir = sys.argv[3] if len(sys.argv) > 3 else 'output_csr'
            convert_hdf5_to_abacus_csr(hdf5_file, output_dir)
        elif sys.argv[2] == '--compare':
            # 对比模式（转换后与原始对比）
            if len(sys.argv) < 5:
                print("Usage: script.py hdf5_file --compare converted_dir original_dir")
                sys.exit(1)
            converted_dir = sys.argv[3]
            original_dir = sys.argv[4]
            compare_csr_files(converted_dir, original_dir)
        else:
            # 查看并对比模式
            original_dir = sys.argv[2]
            print_hdf5_structure(hdf5_file)
            # 可以添加更多对比逻辑
    else:
        # 只打印 HDF5 结构
        print_hdf5_structure(hdf5_file)


if __name__ == '__main__':
    main()
