/*
 * Copyright (c) 2024, UNIVERSITY OF ANTWERP.  All rights reserved.
 *
 * UNIVERSITY OF ANTWERP and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from UNIVERSITY OF ANTWERP is strictly prohibited.
 */

/** @file   testbed_nifti.cu
 *  @author Maximilien Nowak Abdallah, University of antwerp
 */

#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/random_val.cuh> // helpers to generate random values, directions
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/trainable_buffer.cuh>

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/trainer.h>

#include <Rnifti.h>

#include <filesystem/path.h>

#include <fstream>

namespace ngp {

void Testbed::load_nifti(const fs::path& data_path) {
    if (!data_path.exists()) {
        throw std::runtime_error{data_path.str() + " does not exist."};
    }

    tlog::info() << "Loading NIFTI file from " << data_path;
    
    // Load NIFTI file
    nifti_image* nii = nifti_image_read(data_path.str().c_str(), 1);
    if (!nii) {
        throw std::runtime_error{"Failed to load NIFTI file"};
    }
    if (!nii->data) {
        nifti_image_free(nii);
        throw std::runtime_error{"NIFTI file contains no data"};
    }

    // Calculate grid size and allocate temporary CPU memory
    size_t grid_size = nii->nx * nii->ny * nii->nz * sizeof(float);
    std::vector<char> cpugrid(grid_size);
    float* grid_data = reinterpret_cast<float*>(cpugrid.data());

    // Convert NIFTI data to float and find max value for majorant
    float mx = std::numeric_limits<float>::lowest();
    
    switch (nii->datatype) {
        case DT_UINT8: {
            uint8_t* src = (uint8_t*)nii->data;
            for (size_t i = 0; i < nii->nvox; i++) {
                grid_data[i] = static_cast<float>(src[i]);
                mx = std::max(mx, grid_data[i]);
            }
            break;
        }
        case DT_INT16: {
            int16_t* src = (int16_t*)nii->data;
            for (size_t i = 0; i < nii->nvox; i++) {
                grid_data[i] = static_cast<float>(src[i]);
                mx = std::max(mx, grid_data[i]);
            }
            break;
        }
        case DT_FLOAT32: {
            float* src = (float*)nii->data;
            for (size_t i = 0; i < nii->nvox; i++) {
                grid_data[i] = src[i];
                mx = std::max(mx, grid_data[i]);
            }
            break;
        }
        default:
            nifti_image_free(nii);
            throw std::runtime_error{"Unsupported NIFTI datatype"};
    }

    // Copy grid to GPU
    m_volume.nanovdb_grid.enlarge(grid_size);
    m_volume.nanovdb_grid.copy_from_host(cpugrid.data(), grid_size);

    // Set up volume transformation parameters
    int xsize = std::max(1, nii->nx);
    int ysize = std::max(1, nii->ny);
    int zsize = std::max(1, nii->nz);
    float maxsize = std::max(std::max(xsize, ysize), zsize);
    float scale = 1.0f / maxsize;

    // Set bounding boxes
    m_aabb = m_render_aabb = BoundingBox{
        vec3{0.5f - xsize * scale * 0.5f, 0.5f - ysize * scale * 0.5f, 0.5f - zsize * scale * 0.5f},
        vec3{0.5f + xsize * scale * 0.5f, 0.5f + ysize * scale * 0.5f, 0.5f + zsize * scale * 0.5f},
    };
    m_render_aabb_to_local = mat3::identity();

    // Set up world2index transformations
    m_volume.world2index_scale = maxsize;
    m_volume.world2index_offset = vec3{
        xsize * 0.5f - 0.5f * maxsize,
        ysize * 0.5f - 0.5f * maxsize,
        zsize * 0.5f - 0.5f * maxsize
    };

    // Create bitgrid for acceleration
    std::vector<uint8_t> bitgrid;
    bitgrid.resize(128 * 128 * 128 / 8);

    // Fill bitgrid
    for (int x = 0; x < xsize; ++x) {
        for (int y = 0; y < ysize; ++y) {
            for (int z = 0; z < zsize; ++z) {
                float v = grid_data[x + y * xsize + z * xsize * ysize];
                if (v > 0.001f) {
                    float fx = ((x + 0.5f) - m_volume.world2index_offset.x) / m_volume.world2index_scale;
                    float fy = ((y + 0.5f) - m_volume.world2index_offset.y) / m_volume.world2index_scale;
                    float fz = ((z + 0.5f) - m_volume.world2index_offset.z) / m_volume.world2index_scale;
                    uint32_t bitidx = morton3D(int(fx * 128.0f + 0.5f), int(fy * 128.0f + 0.5f), int(fz * 128.0f + 0.5f));
                    if (bitidx < 128 * 128 * 128)
                        bitgrid[bitidx / 8] |= 1 << (bitidx & 7);
                }
            }
        }
    }

    // Copy bitgrid to GPU
    m_volume.bitgrid.enlarge(bitgrid.size());
    m_volume.bitgrid.copy_from_host(bitgrid.data());
    
    // Set global majorant
    m_volume.global_majorant = mx;

    // Clean up
    nifti_image_free(nii);

    tlog::info() << "NIFTI loaded: size=[" << xsize << "," << ysize << "," << zsize 
                 << "], voxel size=[" << nii->dx << "," << nii->dy << "," << nii->dz 
                 << "], global_majorant=" << mx;
}

}
