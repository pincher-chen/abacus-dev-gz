#!/usr/bin/env python3
"""
Read and display information from ABACUS HDF5 matrix output files.

Usage:
    python read_hdf5_matrix.py data-HSR_step0.h5
    python read_hdf5_matrix.py data-HSR_step0.h5 --dataset /hamiltonian/H_R_spin0_R0
"""

import h5py
import numpy as np
import sys
import argparse


def print_h5_structure(name, obj):
    """Callback function to print HDF5 structure."""
    indent = "  " * name.count('/')
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}📄 Dataset: {name}")
        print(f"{indent}   Shape: {obj.shape}, Dtype: {obj.dtype}")
        if obj.attrs:
            print(f"{indent}   Attributes:")
            for key, val in obj.attrs.items():
                print(f"{indent}     - {key}: {val}")
    elif isinstance(obj, h5py.Group):
        print(f"{indent}📁 Group: {name}")
        if obj.attrs:
            print(f"{indent}   Attributes:")
            for key, val in obj.attrs.items():
                print(f"{indent}     - {key}: {val}")


def read_csr_from_hdf5_group(group):
    """
    Read CSR format sparse matrix from HDF5 group.
    
    Args:
        group: HDF5 group containing 'values', 'col_indices', 'row_ptr'
    
    Returns:
        tuple: (values, col_indices, row_ptr, metadata)
    """
    values = group['values'][:]
    col_indices = group['col_indices'][:]
    row_ptr = group['row_ptr'][:]
    
    metadata = {}
    for key in group.attrs.keys():
        metadata[key] = group.attrs[key]
    
    return values, col_indices, row_ptr, metadata


def csr_to_dense(values, col_indices, row_ptr, dimension):
    """Convert CSR format to dense matrix."""
    matrix = np.zeros((dimension, dimension), dtype=values.dtype)
    for i in range(dimension):
        for j in range(row_ptr[i], row_ptr[i+1]):
            matrix[i, col_indices[j]] = values[j]
    return matrix


def analyze_matrix(values, col_indices, row_ptr, name):
    """Analyze and print statistics of a sparse matrix."""
    nnz = len(values)
    dimension = len(row_ptr) - 1
    
    print(f"\n{'='*70}")
    print(f"Matrix: {name}")
    print(f"{'='*70}")
    print(f"  Dimension: {dimension} × {dimension}")
    print(f"  Non-zero elements: {nnz:,}")
    print(f"  Sparsity: {(1 - nnz/(dimension*dimension))*100:.2f}%")
    
    if len(values) > 0:
        if values.dtype == np.complex128:
            abs_values = np.abs(values)
            print(f"  Value range (magnitude):")
            print(f"    Min: {abs_values.min():.6e}")
            print(f"    Max: {abs_values.max():.6e}")
            print(f"    Mean: {abs_values.mean():.6e}")
        else:
            print(f"  Value range:")
            print(f"    Min: {values.min():.6e}")
            print(f"    Max: {values.max():.6e}")
            print(f"    Mean: {values.mean():.6e}")
    
    # Row sparsity pattern
    row_nnz = np.diff(row_ptr)
    print(f"  Elements per row:")
    print(f"    Min: {row_nnz.min()}")
    print(f"    Max: {row_nnz.max()}")
    print(f"    Mean: {row_nnz.mean():.1f}")


def main():
    parser = argparse.ArgumentParser(description='Read ABACUS HDF5 matrix files')
    parser.add_argument('filename', help='HDF5 file to read')
    parser.add_argument('--dataset', '-d', help='Specific dataset to analyze')
    parser.add_argument('--show-structure', '-s', action='store_true',
                       help='Show complete file structure')
    parser.add_argument('--export-dense', '-e', metavar='OUTPUT',
                       help='Export dataset as dense matrix to .npy file')
    
    args = parser.parse_args()
    
    try:
        with h5py.File(args.filename, 'r') as f:
            print(f"\n{'='*70}")
            print(f"HDF5 File: {args.filename}")
            print(f"{'='*70}")
            
            # Show file structure
            if args.show_structure or not args.dataset:
                print("\nFile Structure:")
                f.visititems(print_h5_structure)
            
            # Read metadata
            if '/metadata' in f:
                print(f"\n{'='*70}")
                print("Metadata:")
                print(f"{'='*70}")
                meta = f['/metadata']
                for key, val in meta.attrs.items():
                    print(f"  {key}: {val}")
            
            # Read R vectors
            if '/R_vectors/coordinates' in f:
                R_coords = f['/R_vectors/coordinates'][:]
                print(f"\n{'='*70}")
                print(f"R Vectors: {len(R_coords)} vectors")
                print(f"{'='*70}")
                print("  First 5 R vectors:")
                for i, R in enumerate(R_coords[:5]):
                    print(f"    R[{i}] = ({R[0]:3d}, {R[1]:3d}, {R[2]:3d})")
                if len(R_coords) > 5:
                    print(f"    ... and {len(R_coords)-5} more")
            
            # Analyze specific dataset
            if args.dataset:
                if args.dataset in f:
                    print(f"\n{'='*70}")
                    print(f"Analyzing: {args.dataset}")
                    print(f"{'='*70}")
                    
                    group = f[args.dataset]
                    if isinstance(group, h5py.Group):
                        values, col_indices, row_ptr, metadata = read_csr_from_hdf5_group(group)
                        analyze_matrix(values, col_indices, row_ptr, args.dataset)
                        
                        # Export to dense if requested
                        if args.export_dense:
                            dimension = metadata['dimension']
                            dense_matrix = csr_to_dense(values, col_indices, row_ptr, dimension)
                            np.save(args.export_dense, dense_matrix)
                            print(f"\n✅ Exported dense matrix to {args.export_dense}")
                    else:
                        print(f"  Dataset shape: {group.shape}")
                        print(f"  Dataset dtype: {group.dtype}")
                        print(f"  Data:\n{group[:]}")
                else:
                    print(f"\n❌ Error: Dataset '{args.dataset}' not found in file")
                    sys.exit(1)
            
            # Summary of all matrices
            if not args.dataset:
                print(f"\n{'='*70}")
                print("Matrix Summary:")
                print(f"{'='*70}")
                
                # Count Hamiltonian matrices
                ham_count = 0
                for key in f['/hamiltonian'].keys():
                    if key.startswith('H_R_'):
                        ham_count += 1
                print(f"  Hamiltonian matrices: {ham_count}")
                
                # Count Overlap matrices
                overlap_count = 0
                for key in f['/overlap'].keys():
                    if key.startswith('S_R_'):
                        overlap_count += 1
                print(f"  Overlap matrices: {overlap_count}")
                
                # Estimate file compression
                total_nnz = 0
                for group_name in ['/hamiltonian', '/overlap']:
                    if group_name in f:
                        for key in f[group_name].keys():
                            dataset = f[f'{group_name}/{key}']
                            if isinstance(dataset, h5py.Group) and 'values' in dataset:
                                total_nnz += len(dataset['values'])
                
                if total_nnz > 0:
                    import os
                    file_size_mb = os.path.getsize(args.filename) / 1024 / 1024
                    uncompressed_estimate = total_nnz * 8 / 1024 / 1024  # 8 bytes per double
                    print(f"\n  File size: {file_size_mb:.2f} MB")
                    print(f"  Estimated uncompressed: {uncompressed_estimate:.2f} MB")
                    print(f"  Compression ratio: {(1 - file_size_mb/uncompressed_estimate)*100:.1f}%")
            
            print()  # Final newline
            
    except FileNotFoundError:
        print(f"❌ Error: File '{args.filename}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading file: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

