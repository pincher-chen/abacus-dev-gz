#ifdef __USEHDF5

#include "write_HS_R_hdf5.h"
#include "module_base/parallel_reduce.h"
#include "module_parameter/parameter.h"
#include "module_base/global_function.h"
#include "module_base/global_variable.h"
#include "module_base/tool_title.h"
#include "module_hamilt_lcao/hamilt_lcaodft/spar_hsr.h"
#include <complex>
#include <vector>
#include <map>
#include <iostream>

namespace ModuleIO
{

// Helper function to create HDF5 group
static hid_t create_or_open_group(hid_t loc_id, const char* group_name)
{
    H5E_BEGIN_TRY {
        hid_t group_id = H5Gopen(loc_id, group_name, H5P_DEFAULT);
        if (group_id >= 0) {
            return group_id;
        }
    } H5E_END_TRY;
    
    return H5Gcreate(loc_id, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
}

// Helper function to write a scalar attribute
template<typename T>
static void write_scalar_attribute(hid_t loc_id, const char* attr_name, T value, hid_t type_id)
{
    hid_t dataspace_id = H5Screate(H5S_SCALAR);
    hid_t attr_id = H5Acreate(loc_id, attr_name, type_id, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr_id, type_id, &value);
    H5Aclose(attr_id);
    H5Sclose(dataspace_id);
}

/**
 * @brief 写入单个 R 向量的稀疏矩阵到 HDF5
 * 
 * 此函数会做 MPI 归约，收集所有进程的数据后写入 HDF5
 * 即使矩阵为空（nnz=0），也会创建 group 并存储元数据
 * 
 * @param file_id HDF5 文件 ID
 * @param dataset_name 数据集名称
 * @param XR 稀疏矩阵数据 (本地数据)
 * @param sparse_threshold 稀疏阈值
 * @param compression_level 压缩级别
 * @param pv 并行轨道信息
 */
template<typename T>
void write_sparse_matrix_hdf5_mpi(hid_t file_id,
                                   const std::string& dataset_name,
                                   const std::map<size_t, std::map<size_t, T>>& XR,
                                   const double& sparse_threshold,
                                   const int compression_level,
                                   const Parallel_Orbitals& pv)
{
    const int nlocal = PARAM.globalv.nlocal;
    
    // 分配临时数组用于 MPI 归约
    T* line = new T[nlocal];
    
    // 收集数据并构建 CSR 格式
    std::vector<T> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
    row_ptr.reserve(nlocal + 1);
    row_ptr.push_back(0);
    
    for (int row = 0; row < nlocal; ++row)
    {
        ModuleBase::GlobalFunc::ZEROS(line, nlocal);
        
        // 只有拥有这一行的进程填充数据
        if (pv.global2local_row(row) >= 0)
        {
            auto iter = XR.find(row);
            if (iter != XR.end())
            {
                for (const auto& col_val : iter->second)
                {
                    line[col_val.first] = col_val.second;
                }
            }
        }
        
        // MPI 归约：收集所有进程的数据
        Parallel_Reduce::reduce_all(line, nlocal);
        
        // 只在 rank 0 处理数据
        if (GlobalV::DRANK == 0)
        {
            for (int col = 0; col < nlocal; ++col)
            {
                if (std::abs(line[col]) > sparse_threshold)
                {
                    values.push_back(line[col]);
                    col_indices.push_back(col);
                }
            }
            row_ptr.push_back(static_cast<int>(values.size()));
        }
    }
    
    delete[] line;
    
    // 只在 rank 0 写入 HDF5
    if (GlobalV::DRANK != 0)
    {
        return;
    }
    
    size_t nnz = values.size();
    
    // 创建 group（即使 nnz=0 也创建）
    hid_t group_id = create_or_open_group(file_id, dataset_name.c_str());
    
    // 写入元数据 attributes
    write_scalar_attribute(group_id, "dimension", nlocal, H5T_NATIVE_INT);
    write_scalar_attribute(group_id, "nnz", static_cast<long long>(nnz), H5T_NATIVE_LLONG);
    write_scalar_attribute(group_id, "sparse_threshold", sparse_threshold, H5T_NATIVE_DOUBLE);
    
    // 如果 nnz=0，只存储元数据，不创建数据 dataset
    if (nnz == 0)
    {
        H5Gclose(group_id);
        return;
    }
    
    // Set up compression
    hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    if (compression_level > 0 && compression_level <= 9)
    {
        hsize_t chunk_dims[1];
        chunk_dims[0] = std::min(static_cast<hsize_t>(nnz), static_cast<hsize_t>(10000));
        H5Pset_chunk(dcpl_id, 1, chunk_dims);
        H5Pset_deflate(dcpl_id, compression_level);
    }
    
    // Determine HDF5 data type
    hid_t value_type_id;
    bool is_complex = false;
    if (std::is_same<T, double>::value)
    {
        value_type_id = H5T_NATIVE_DOUBLE;
    }
    else if (std::is_same<T, std::complex<double>>::value)
    {
        is_complex = true;
        value_type_id = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<double>));
        H5Tinsert(value_type_id, "real", 0, H5T_NATIVE_DOUBLE);
        H5Tinsert(value_type_id, "imag", sizeof(double), H5T_NATIVE_DOUBLE);
    }
    else
    {
        H5Pclose(dcpl_id);
        H5Gclose(group_id);
        throw std::runtime_error("Unsupported data type for HDF5 output");
    }
    
    // Write values
    hsize_t dims[1] = {nnz};
    hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
    hid_t dataset_id = H5Dcreate(group_id, "values", value_type_id, dataspace_id,
                                  H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
    H5Dwrite(dataset_id, value_type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    
    // Write col_indices
    dims[0] = nnz;
    dataspace_id = H5Screate_simple(1, dims, NULL);
    dataset_id = H5Dcreate(group_id, "col_indices", H5T_NATIVE_INT, dataspace_id,
                           H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, col_indices.data());
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    
    // Write row_ptr (不压缩这个小数组)
    dims[0] = nlocal + 1;
    dataspace_id = H5Screate_simple(1, dims, NULL);
    dataset_id = H5Dcreate(group_id, "row_ptr", H5T_NATIVE_INT, dataspace_id,
                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, row_ptr.data());
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    
    // Clean up
    if (is_complex)
    {
        H5Tclose(value_type_id);
    }
    H5Pclose(dcpl_id);
    H5Gclose(group_id);
}

// Explicit template instantiation
template void write_sparse_matrix_hdf5_mpi<double>(hid_t, const std::string&,
    const std::map<size_t, std::map<size_t, double>>&, const double&, const int, const Parallel_Orbitals&);
template void write_sparse_matrix_hdf5_mpi<std::complex<double>>(hid_t, const std::string&,
    const std::map<size_t, std::map<size_t, std::complex<double>>>&, const double&, const int, const Parallel_Orbitals&);

// 主函数：输出 H(R) 和 S(R) 到 HDF5 格式
void output_HSR_hdf5(const UnitCell& ucell,
                     const int& istep,
                     const ModuleBase::matrix& v_eff,
                     const Parallel_Orbitals& pv,
                     LCAO_HS_Arrays& HS_Arrays,
                     const Grid_Driver& grid,
                     const K_Vectors& kv,
                     hamilt::Hamilt<std::complex<double>>* p_ham,
#ifdef __EXX
                     const std::vector<std::map<int, std::map<TAC, RI::Tensor<double>>>>* Hexxd,
                     const std::vector<std::map<int, std::map<TAC, RI::Tensor<std::complex<double>>>>>* Hexxc,
#endif
                     const std::string& hdf5_filename,
                     const double& sparse_threshold,
                     const int compression_level)
{
    ModuleBase::TITLE("ModuleIO", "output_HSR_hdf5");
    ModuleBase::timer::tick("ModuleIO", "output_HSR_hdf5");

    const int nspin = PARAM.inp.nspin;
    const int nlocal = PARAM.globalv.nlocal;

    // 使用 sparse_format::cal_HSR 填充 HS_Arrays
    if (nspin == 1 || nspin == 4)
    {
        const int spin_now = 0;
        sparse_format::cal_HSR(ucell, pv, HS_Arrays, grid, spin_now, sparse_threshold, kv.nmp, p_ham
#ifdef __EXX
            , Hexxd, Hexxc
#endif
        );
    }
    else if (nspin == 2)
    {
        int spin_now = 1;
        sparse_format::cal_HSR(ucell, pv, HS_Arrays, grid, spin_now, sparse_threshold, kv.nmp, p_ham
#ifdef __EXX
            , Hexxd, Hexxc
#endif
        );

        if (PARAM.inp.vl_in_h)
        {
            const int ik = 0;
            p_ham->refresh();
            p_ham->updateHk(ik);
            spin_now = 0;
        }

        sparse_format::cal_HSR(ucell, pv, HS_Arrays, grid, spin_now, sparse_threshold, kv.nmp, p_ham
#ifdef __EXX
            , Hexxd, Hexxc
#endif
        );
    }

    // 获取所有 R 向量
    auto& all_R_coor = HS_Arrays.all_R_coor;
    auto& HR_sparse = HS_Arrays.HR_sparse;
    auto& SR_sparse = HS_Arrays.SR_sparse;
    
    int spin_loop = (nspin == 2) ? 2 : 1;
    int total_R_num = all_R_coor.size();
    
    // 统计非零元素数量（需要 MPI 归约）
    int* H_nonzero_num[2] = {nullptr, nullptr};
    int* S_nonzero_num = new int[total_R_num];
    ModuleBase::GlobalFunc::ZEROS(S_nonzero_num, total_R_num);
    
    for (int ispin = 0; ispin < spin_loop; ++ispin)
    {
        H_nonzero_num[ispin] = new int[total_R_num];
        ModuleBase::GlobalFunc::ZEROS(H_nonzero_num[ispin], total_R_num);
    }
    
    int count = 0;
    for (const auto& R_coor : all_R_coor)
    {
        if (nspin != 4)
        {
            for (int ispin = 0; ispin < spin_loop; ++ispin)
            {
                auto iter = HR_sparse[ispin].find(R_coor);
                if (iter != HR_sparse[ispin].end())
                {
                    for (const auto& row_loop : iter->second)
                    {
                        H_nonzero_num[ispin][count] += row_loop.second.size();
                    }
                }
            }
            
            auto iter = SR_sparse.find(R_coor);
            if (iter != SR_sparse.end())
            {
                for (const auto& row_loop : iter->second)
                {
                    S_nonzero_num[count] += row_loop.second.size();
                }
            }
        }
        count++;
    }
    
    // MPI 归约非零元素计数
    Parallel_Reduce::reduce_all(S_nonzero_num, total_R_num);
    for (int ispin = 0; ispin < spin_loop; ++ispin)
    {
        Parallel_Reduce::reduce_all(H_nonzero_num[ispin], total_R_num);
    }
    
    // 确定哪些 R 向量应该输出（与原始 save_HSR_sparse 逻辑一致）
    // 只有当 H 或 S 至少有一个矩阵在该 R 向量上有非零元素时，才输出
    // 使用 vector<bool> 而不是 set，避免 set::find 的潜在问题
    std::vector<bool> should_output(total_R_num, false);
    int output_R_number = 0;
    
    for (int idx = 0; idx < total_R_num; ++idx)
    {
        bool has_nonzero = false;
        
        if (nspin == 2)
        {
            if (H_nonzero_num[0][idx] != 0 || H_nonzero_num[1][idx] != 0 
                || S_nonzero_num[idx] != 0)
            {
                has_nonzero = true;
            }
        }
        else
        {
            if (H_nonzero_num[0][idx] != 0 || S_nonzero_num[idx] != 0)
            {
                has_nonzero = true;
            }
        }
        
        if (has_nonzero)
        {
            should_output[idx] = true;
            output_R_number++;
        }
    }
    
    // 只在 rank 0 创建 HDF5 文件
    hid_t file_id = -1;
    if (GlobalV::DRANK == 0)
    {
        std::string filename = PARAM.globalv.global_out_dir + hdf5_filename;
        file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        
        if (file_id < 0)
        {
            std::cerr << "Error: Cannot create HDF5 file: " << filename << std::endl;
        }
        else
        {
            std::cout << "Writing matrices to HDF5 file: " << filename << std::endl;
            std::cout << "  Compression level: " << compression_level << std::endl;
            std::cout << "  Output R vectors: " << output_R_number << " (total: " << total_R_num << ")" << std::endl;
            std::cout << "  Matrix dimension: " << nlocal << std::endl;
            
            // Write metadata
            hid_t meta_group_id = create_or_open_group(file_id, "/metadata");
            write_scalar_attribute(meta_group_id, "step", istep, H5T_NATIVE_INT);
            write_scalar_attribute(meta_group_id, "nspin", nspin, H5T_NATIVE_INT);
            write_scalar_attribute(meta_group_id, "dimension", nlocal, H5T_NATIVE_INT);
            write_scalar_attribute(meta_group_id, "num_R_vectors", output_R_number, H5T_NATIVE_INT);
            H5Gclose(meta_group_id);
            
            // Create groups
            hid_t hamil_group_id = create_or_open_group(file_id, "/hamiltonian");
            hid_t overlap_group_id = create_or_open_group(file_id, "/overlap");
            H5Gclose(hamil_group_id);
            H5Gclose(overlap_group_id);
            
            // Write output R vectors
            hid_t R_group_id = create_or_open_group(file_id, "/R_vectors");
            std::vector<int> R_flat;
            R_flat.reserve(output_R_number * 3);
            
            int r_idx = 0;
            for (const auto& R : all_R_coor)
            {
                if (should_output[r_idx])
                {
                    R_flat.push_back(R.x);
                    R_flat.push_back(R.y);
                    R_flat.push_back(R.z);
                }
                r_idx++;
            }
            
            if (!R_flat.empty())
            {
                hsize_t dims[2] = {static_cast<hsize_t>(output_R_number), 3};
                hid_t dataspace_id = H5Screate_simple(2, dims, NULL);
                hid_t dataset_id = H5Dcreate(R_group_id, "coordinates", H5T_NATIVE_INT, dataspace_id,
                                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, R_flat.data());
                H5Dclose(dataset_id);
                H5Sclose(dataspace_id);
            }
            H5Gclose(R_group_id);
        }
    }
    
    // 广播 file_id 有效性
    int file_valid = (file_id >= 0) ? 1 : 0;
    Parallel_Reduce::reduce_all(file_valid);
    
    if (file_valid == 0)
    {
        for (int ispin = 0; ispin < spin_loop; ++ispin)
        {
            delete[] H_nonzero_num[ispin];
        }
        delete[] S_nonzero_num;
        sparse_format::destroy_HS_R_sparse(HS_Arrays);
        ModuleBase::timer::tick("ModuleIO", "output_HSR_hdf5");
        return;
    }
    
    // 写入 S(R) 和 H(R) 矩阵（只输出 should_output[i]=true 的 R 向量）
    if (GlobalV::DRANK == 0)
    {
        std::cout << "  Writing overlap matrix S(R)..." << std::endl;
    }
    
    int R_index = 0;
    int S_nonzero_count = 0;
    count = 0;
    
    for (const auto& R_coor : all_R_coor)
    {
        // 检查这个 R 向量是否应该输出
        if (!should_output[count])
        {
            count++;
            continue;
        }
        
        // 写入 S(R)
        std::string S_dataset_name = "/overlap/S_R_" + std::to_string(R_index);
        
        auto S_iter = SR_sparse.find(R_coor);
        if (S_iter != SR_sparse.end())
        {
            write_sparse_matrix_hdf5_mpi(file_id, S_dataset_name, S_iter->second,
                                         sparse_threshold, compression_level, pv);
            if (S_nonzero_num[count] > 0)
            {
                S_nonzero_count++;
            }
        }
        else
        {
            // 空矩阵
            std::map<size_t, std::map<size_t, double>> empty_map;
            write_sparse_matrix_hdf5_mpi(file_id, S_dataset_name, empty_map,
                                         sparse_threshold, compression_level, pv);
        }
        
        R_index++;
        count++;
    }
    
    if (GlobalV::DRANK == 0)
    {
        std::cout << "    Total S(R) matrices: " << R_index 
                  << " (non-zero: " << S_nonzero_count << ")" << std::endl;
    }
    
    // 写入 H(R)
    for (int ispin = 0; ispin < spin_loop; ++ispin)
    {
        if (GlobalV::DRANK == 0)
        {
            std::cout << "  Writing Hamiltonian H(R) for spin " << ispin << "..." << std::endl;
        }
        
        R_index = 0;
        int H_nonzero_count = 0;
        count = 0;
        
        for (const auto& R_coor : all_R_coor)
        {
            // 检查这个 R 向量是否应该输出
            if (!should_output[count])
            {
                count++;
                continue;
            }
            
            std::string H_dataset_name = "/hamiltonian/H_R_spin" + std::to_string(ispin) 
                                       + "_R" + std::to_string(R_index);
            
            auto H_iter = HR_sparse[ispin].find(R_coor);
            if (H_iter != HR_sparse[ispin].end())
            {
                write_sparse_matrix_hdf5_mpi(file_id, H_dataset_name, H_iter->second,
                                             sparse_threshold, compression_level, pv);
                if (H_nonzero_num[ispin][count] > 0)
                {
                    H_nonzero_count++;
                }
            }
            else
            {
                // 空矩阵
                std::map<size_t, std::map<size_t, double>> empty_map;
                write_sparse_matrix_hdf5_mpi(file_id, H_dataset_name, empty_map,
                                             sparse_threshold, compression_level, pv);
            }
            
            R_index++;
            count++;
        }
        
        if (GlobalV::DRANK == 0)
        {
            std::cout << "    Total H(R) matrices for spin " << ispin << ": " << R_index
                      << " (non-zero: " << H_nonzero_count << ")" << std::endl;
        }
    }
    
    // 关闭 HDF5 文件
    if (GlobalV::DRANK == 0 && file_id >= 0)
    {
        H5Fclose(file_id);
        std::cout << "Successfully wrote HDF5 file with " << output_R_number << " R vectors." << std::endl;
    }
    
    // 清理
    for (int ispin = 0; ispin < spin_loop; ++ispin)
    {
        delete[] H_nonzero_num[ispin];
    }
    delete[] S_nonzero_num;
    sparse_format::destroy_HS_R_sparse(HS_Arrays);
    
    ModuleBase::timer::tick("ModuleIO", "output_HSR_hdf5");
}

// 保留旧的模板函数用于兼容性（不带 MPI 归约）
template<typename T>
void write_sparse_matrix_hdf5(hid_t file_id,
                               const std::string& dataset_name,
                               const std::map<size_t, std::map<size_t, T>>& XR,
                               const int dimension,
                               const double& sparse_threshold,
                               const int compression_level)
{
    size_t nnz = 0;
    for (const auto& row_pair : XR) {
        for (const auto& col_pair : row_pair.second) {
            if (std::abs(col_pair.second) > sparse_threshold) {
                nnz++;
            }
        }
    }

    hid_t group_id = create_or_open_group(file_id, dataset_name.c_str());
    
    // 写入元数据
    write_scalar_attribute(group_id, "dimension", dimension, H5T_NATIVE_INT);
    write_scalar_attribute(group_id, "nnz", static_cast<long long>(nnz), H5T_NATIVE_LLONG);
    write_scalar_attribute(group_id, "sparse_threshold", sparse_threshold, H5T_NATIVE_DOUBLE);

    if (nnz == 0) {
        H5Gclose(group_id);
        return;
    }

    std::vector<T> values(nnz);
    std::vector<int> col_indices(nnz);
    std::vector<int> row_ptr(dimension + 1, 0);

    size_t idx = 0;
    for (int row = 0; row < dimension; ++row) {
        row_ptr[row] = idx;
        auto row_it = XR.find(row);
        if (row_it != XR.end()) {
            for (const auto& col_pair : row_it->second) {
                if (std::abs(col_pair.second) > sparse_threshold) {
                    values[idx] = col_pair.second;
                    col_indices[idx] = col_pair.first;
                    idx++;
                }
            }
        }
    }
    row_ptr[dimension] = idx;

    hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    if (compression_level > 0 && compression_level <= 9) {
        hsize_t chunk_dims[1];
        chunk_dims[0] = std::min(static_cast<hsize_t>(nnz), static_cast<hsize_t>(10000));
        H5Pset_chunk(dcpl_id, 1, chunk_dims);
        H5Pset_deflate(dcpl_id, compression_level);
    }

    hsize_t dims[1] = {nnz};
    hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
    
    hid_t value_type_id;
    bool is_complex = false;
    if (std::is_same<T, double>::value) {
        value_type_id = H5T_NATIVE_DOUBLE;
    } else if (std::is_same<T, std::complex<double>>::value) {
        is_complex = true;
        value_type_id = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<double>));
        H5Tinsert(value_type_id, "real", 0, H5T_NATIVE_DOUBLE);
        H5Tinsert(value_type_id, "imag", sizeof(double), H5T_NATIVE_DOUBLE);
    } else {
        H5Pclose(dcpl_id);
        H5Gclose(group_id);
        throw std::runtime_error("Unsupported data type for HDF5 output");
    }

    hid_t dataset_id = H5Dcreate(group_id, "values", value_type_id, dataspace_id, 
                                  H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
    H5Dwrite(dataset_id, value_type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    dims[0] = nnz;
    dataspace_id = H5Screate_simple(1, dims, NULL);
    dataset_id = H5Dcreate(group_id, "col_indices", H5T_NATIVE_INT, dataspace_id,
                           H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, col_indices.data());
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    dims[0] = dimension + 1;
    dataspace_id = H5Screate_simple(1, dims, NULL);
    dataset_id = H5Dcreate(group_id, "row_ptr", H5T_NATIVE_INT, dataspace_id,
                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, row_ptr.data());
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    if (is_complex) {
        H5Tclose(value_type_id);
    }
    H5Pclose(dcpl_id);
    H5Gclose(group_id);
}

template void write_sparse_matrix_hdf5<double>(hid_t, const std::string&, 
    const std::map<size_t, std::map<size_t, double>>&, const int, const double&, const int);
template void write_sparse_matrix_hdf5<std::complex<double>>(hid_t, const std::string&,
    const std::map<size_t, std::map<size_t, std::complex<double>>>&, const int, const double&, const int);

} // namespace ModuleIO

#endif // __USEHDF5
