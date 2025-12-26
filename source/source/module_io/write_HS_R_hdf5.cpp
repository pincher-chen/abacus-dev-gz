#ifdef __USEHDF5

#include "write_HS_R_hdf5.h"
#include "module_base/parallel_reduce.h"
#include "module_parameter/parameter.h"
#include "module_base/global_function.h"
#include "module_base/global_variable.h"
#include "module_base/tool_title.h"
#include "write_HS.hpp"
#include <complex>
#include <vector>
#include <map>
#include <iostream>

namespace ModuleIO
{

// Helper function to create HDF5 group
static hid_t create_or_open_group(hid_t loc_id, const char* group_name)
{
    // Try to open existing group
    H5E_BEGIN_TRY {
        hid_t group_id = H5Gopen(loc_id, group_name, H5P_DEFAULT);
        if (group_id >= 0) {
            return group_id;
        }
    } H5E_END_TRY;
    
    // Create new group
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

// Template specialization for writing sparse matrices to HDF5
template<typename T>
void write_sparse_matrix_hdf5(hid_t file_id,
                               const std::string& dataset_name,
                               const std::map<size_t, std::map<size_t, T>>& XR,
                               const int dimension,
                               const double& sparse_threshold,
                               const int compression_level)
{
    // Count non-zero elements
    size_t nnz = 0;
    for (const auto& row_pair : XR) {
        for (const auto& col_pair : row_pair.second) {
            if (std::abs(col_pair.second) > sparse_threshold) {
                nnz++;
            }
        }
    }

    if (nnz == 0) {
        std::cout << "Warning: Matrix " << dataset_name << " has no non-zero elements." << std::endl;
        return;
    }

    // Allocate arrays for CSR format
    std::vector<T> values(nnz);
    std::vector<int> col_indices(nnz);
    std::vector<int> row_ptr(dimension + 1, 0);

    // Fill CSR arrays
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

    // Create group for this matrix
    hid_t group_id = create_or_open_group(file_id, dataset_name.c_str());

    // Set up compression
    hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    if (compression_level > 0 && compression_level <= 9) {
        hsize_t chunk_dims[1];
        
        // Chunk size optimization
        chunk_dims[0] = std::min(static_cast<hsize_t>(nnz), static_cast<hsize_t>(10000));
        H5Pset_chunk(dcpl_id, 1, chunk_dims);
        H5Pset_deflate(dcpl_id, compression_level);
    }

    // Write values
    hsize_t dims[1] = {nnz};
    hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
    
    // Determine HDF5 data type based on T
    hid_t value_type_id;
    if (std::is_same<T, double>::value) {
        value_type_id = H5T_NATIVE_DOUBLE;
    } else if (std::is_same<T, std::complex<double>>::value) {
        // Create complex double type
        value_type_id = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<double>));
        H5Tinsert(value_type_id, "real", 0, H5T_NATIVE_DOUBLE);
        H5Tinsert(value_type_id, "imag", sizeof(double), H5T_NATIVE_DOUBLE);
    } else {
        throw std::runtime_error("Unsupported data type for HDF5 output");
    }

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

    // Write row_ptr (no compression for this small array)
    dims[0] = dimension + 1;
    dataspace_id = H5Screate_simple(1, dims, NULL);
    dataset_id = H5Dcreate(group_id, "row_ptr", H5T_NATIVE_INT, dataspace_id,
                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, row_ptr.data());
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    // Write metadata attributes
    write_scalar_attribute(group_id, "dimension", dimension, H5T_NATIVE_INT);
    write_scalar_attribute(group_id, "nnz", static_cast<long long>(nnz), H5T_NATIVE_LLONG);
    write_scalar_attribute(group_id, "sparse_threshold", sparse_threshold, H5T_NATIVE_DOUBLE);

    // Clean up
    if (std::is_same<T, std::complex<double>>::value) {
        H5Tclose(value_type_id);
    }
    H5Pclose(dcpl_id);
    H5Gclose(group_id);
}

// Explicit template instantiation
template void write_sparse_matrix_hdf5<double>(hid_t, const std::string&, 
    const std::map<size_t, std::map<size_t, double>>&, const int, const double&, const int);
template void write_sparse_matrix_hdf5<std::complex<double>>(hid_t, const std::string&,
    const std::map<size_t, std::map<size_t, std::complex<double>>>&, const int, const double&, const int);

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

    if (GlobalV::DRANK != 0) {
        ModuleBase::timer::tick("ModuleIO", "output_HSR_hdf5");
        return; // Only rank 0 writes the HDF5 file
    }

    // Construct output matrices
    output_HS_R(istep, v_eff, pv, ucell, grid, kv, p_ham, HS_Arrays, sparse_threshold
#ifdef __EXX
                , Hexxd, Hexxc
#endif
    );

    // Create HDF5 file
    std::string filename = PARAM.globalv.global_out_dir + hdf5_filename;
    hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    
    if (file_id < 0) {
        std::cerr << "Error: Cannot create HDF5 file: " << filename << std::endl;
        ModuleBase::timer::tick("ModuleIO", "output_HSR_hdf5");
        return;
    }

    std::cout << "Writing matrices to HDF5 file: " << filename << std::endl;
    std::cout << "  Compression level: " << compression_level << std::endl;

    // Write metadata
    hid_t meta_group_id = create_or_open_group(file_id, "/metadata");
    write_scalar_attribute(meta_group_id, "step", istep, H5T_NATIVE_INT);
    write_scalar_attribute(meta_group_id, "nspin", PARAM.inp.nspin, H5T_NATIVE_INT);
    write_scalar_attribute(meta_group_id, "dimension", PARAM.globalv.nlocal, H5T_NATIVE_INT);
    write_scalar_attribute(meta_group_id, "num_R_vectors", 
                          static_cast<int>(HS_Arrays.output_R_coor.size()), H5T_NATIVE_INT);
    H5Gclose(meta_group_id);

    // Write R vectors
    if (!HS_Arrays.output_R_coor.empty()) {
        hid_t group_id = create_or_open_group(file_id, "/R_vectors");
        
        std::vector<int> R_flat;
        R_flat.reserve(HS_Arrays.output_R_coor.size() * 3);
        for (const auto& R : HS_Arrays.output_R_coor) {
            R_flat.push_back(R.x);
            R_flat.push_back(R.y);
            R_flat.push_back(R.z);
        }
        
        hsize_t dims[2] = {HS_Arrays.output_R_coor.size(), 3};
        hid_t dataspace_id = H5Screate_simple(2, dims, NULL);
        hid_t dataset_id = H5Dcreate(group_id, "coordinates", H5T_NATIVE_INT, dataspace_id,
                                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, R_flat.data());
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        H5Gclose(group_id);
    }

    // Create hamiltonian and overlap groups
    hid_t hamil_group_id = create_or_open_group(file_id, "/hamiltonian");
    hid_t overlap_group_id = create_or_open_group(file_id, "/overlap");

    // Write S(R) - Overlap matrix
    std::cout << "  Writing overlap matrix S(R)..." << std::endl;
    int R_index = 0;
    for (auto& SR_ptr : HS_Arrays.SR_sparse) {
        std::string dataset_name = "/overlap/S_R_" + std::to_string(R_index);
        write_sparse_matrix_hdf5(file_id, dataset_name, SR_ptr,
                                 PARAM.globalv.nlocal, sparse_threshold, compression_level);
        R_index++;
    }

    // Write H(R) - Hamiltonian matrices
    for (int ispin = 0; ispin < PARAM.inp.nspin; ++ispin) {
        std::cout << "  Writing Hamiltonian H(R) for spin " << ispin << "..." << std::endl;
        R_index = 0;
        for (auto& HR_ptr : HS_Arrays.HR_sparse[ispin]) {
            std::string dataset_name = "/hamiltonian/H_R_spin" + std::to_string(ispin) 
                                     + "_R" + std::to_string(R_index);
            write_sparse_matrix_hdf5(file_id, dataset_name, HR_ptr,
                                     PARAM.globalv.nlocal, sparse_threshold, compression_level);
            R_index++;
        }
    }

    // Clean up
    H5Gclose(hamil_group_id);
    H5Gclose(overlap_group_id);
    H5Fclose(file_id);

    std::cout << "Successfully wrote HDF5 file: " << filename << std::endl;
    std::cout << "  Total R vectors: " << HS_Arrays.output_R_coor.size() << std::endl;
    std::cout << "  Matrix dimension: " << PARAM.globalv.nlocal << std::endl;

    ModuleBase::timer::tick("ModuleIO", "output_HSR_hdf5");
}

} // namespace ModuleIO

#endif // __USEHDF5

