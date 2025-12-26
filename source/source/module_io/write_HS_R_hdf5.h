#ifndef WRITE_HS_R_HDF5_H
#define WRITE_HS_R_HDF5_H

#ifdef __USEHDF5

#include "module_base/matrix.h"
#include "module_basis/module_nao/two_center_bundle.h"
#include "module_cell/klist.h"
#include "module_hamilt_general/hamilt.h"
#include "module_hamilt_lcao/module_gint/gint_k.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include <hdf5.h>

namespace ModuleIO
{
    using TAC = std::pair<int, std::array<int, 3>>;
    
    /**
     * @brief Output Hamiltonian and Overlap matrices to HDF5 format
     * 
     * This function exports H(R) and S(R) matrices to an HDF5 file with internal gzip compression.
     * The HDF5 format provides:
     * - Efficient storage with compression (typically 70% reduction)
     * - Fast random access to specific R vectors
     * - Self-describing metadata
     * - Cross-platform compatibility
     * - Support for parallel I/O (with MPI-enabled HDF5)
     * 
     * File structure:
     * /
     * ├── /metadata
     * │   ├── step (scalar int)
     * │   ├── nspin (scalar int)
     * │   ├── dimension (scalar int)
     * │   ├── num_R_vectors (scalar int)
     * │   └── fermi_energy (scalar double)
     * ├── /hamiltonian
     * │   ├── H_R_spin0 (dataset, complex double, compressed)
     * │   ├── H_R_spin1 (dataset, complex double, compressed) [if nspin > 1]
     * │   └── R_vectors (dataset, int)
     * └── /overlap
     *     └── S_R (dataset, complex double, compressed)
     * 
     * @param ucell Unit cell information
     * @param istep Current MD/relaxation step
     * @param v_eff Effective potential
     * @param pv Parallel orbitals
     * @param HS_Arrays Arrays containing H and S matrices
     * @param grid Grid driver
     * @param kv K-point vectors
     * @param p_ham Hamiltonian pointer
     * @param Hexxd Exchange-correlation tensor (double) for EXX
     * @param Hexxc Exchange-correlation tensor (complex) for EXX
     * @param hdf5_filename Output HDF5 filename
     * @param sparse_threshold Threshold for considering matrix elements as zero
     * @param compression_level gzip compression level (0-9, default 6)
     */
    void output_HSR_hdf5(const UnitCell& ucell,
                         const int& istep,
                         const ModuleBase::matrix& v_eff,
                         const Parallel_Orbitals& pv,
                         LCAO_HS_Arrays& HS_Arrays,
                         const Grid_Driver& grid,
                         const K_Vectors& kv,
                         hamilt::Hamilt<std::complex<double>>* p_ham,
#ifdef __EXX
                         const std::vector<std::map<int, std::map<TAC, RI::Tensor<double>>>>* Hexxd = nullptr,
                         const std::vector<std::map<int, std::map<TAC, RI::Tensor<std::complex<double>>>>>* Hexxc = nullptr,
#endif
                         const std::string& hdf5_filename = "data-HSR.h5",
                         const double& sparse_threshold = 1e-10,
                         const int compression_level = 6);

    /**
     * @brief Write a single sparse matrix (H or S) to HDF5 dataset
     * 
     * @param file_id HDF5 file identifier
     * @param dataset_name Name of the dataset to create
     * @param XR Sparse matrix data in CSR-like format
     * @param dimension Matrix dimension
     * @param sparse_threshold Threshold for sparsity
     * @param compression_level gzip compression level
     */
    template<typename T>
    void write_sparse_matrix_hdf5(hid_t file_id,
                                   const std::string& dataset_name,
                                   const std::map<size_t, std::map<size_t, T>>& XR,
                                   const int dimension,
                                   const double& sparse_threshold,
                                   const int compression_level);

} // namespace ModuleIO

#endif // __USEHDF5
#endif // WRITE_HS_R_HDF5_H

