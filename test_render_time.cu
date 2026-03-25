#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define BLOCK_SIZE 256
#define BLOCK_X 256
#define BLOCK_Y 1
#define CHANNELS 3
#define NUM_GAUSSIANS 10000
#define NUM_PIXELS 2473 * 1643
#define WIDTH 2473
#define HEIGHT 1643

__global__ void benchmark_kernel(
    int toDo_initial,
    float* features,
    float* sampled_T,
    float* sampled_ar,
    int* metricCount,
    float* metric_map,
    float* out_dummy,
    bool get_flag,
    float* gt_image,
    float* out_color,
    float* metric_per_gs,
    float* gs_weight,
    // Added arguments
    float* final_T,
    int* n_contrib,
    float* pixel_colors,
    float* bg_color,
    int* max_contrib,
    int* point_list,
    float2* points_xy_image,
    float4* conic_opacity,
    int2 range,
    int rounds,
    long long* debug_cycles
) {
    long long t_start, t_end;
    long long t1 = 0, t2 = 0, t3 = 0;

    // 代码段1
    t_start = clock64();
    auto block = cg::this_thread_block();
    // Shared memory simulation
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
    __shared__ int collected_id[BLOCK_SIZE];

    int tid = threadIdx.x;
    
    // Initialize shared memory
    if (tid < BLOCK_SIZE) {
        collected_xy[tid] = make_float2(100.0f, 100.0f); 
        collected_conic_opacity[tid] = make_float4(1.0f, 0.0f, 1.0f, 0.5f); 
        collected_id[tid] = tid % NUM_GAUSSIANS;
    }
    __syncthreads();

    // Thread local state
    float T = 1.0f;
    float C[CHANNELS] = {0.0f};
    bool done = false;
    int bbm = 0;
    int contributor = 0;
    int last_contributor = 0;
    int contribs = 0;
    float2 pixf = make_float2(100.0f, 100.0f);
    int pix_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure we don't go out of bounds for pixel arrays
    if (pix_id >= NUM_PIXELS) return;

    int toDo = toDo_initial;
    bool inside = true; 

    for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
        if (j % 32 == 0) {
            sampled_T[(bbm * BLOCK_SIZE) + tid] = T;
            for (int ch = 0; ch < CHANNELS; ++ch) {
                sampled_ar[(bbm * BLOCK_SIZE * CHANNELS) + ch * BLOCK_SIZE + tid] = C[ch];
            }
            ++bbm;
        }

        contributor++;

        float2 xy = collected_xy[j];
        float2 d = { xy.x - pixf.x, xy.y - pixf.y };
        float4 con_o = collected_conic_opacity[j];
        float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
        
        if (power > 0.0f)
            continue;

        if (con_o.w >= 0) {
            float alpha = min(0.99f, con_o.w * expf(power));

            if (alpha < 1.0f / 255.0f)
                continue;
            float test_T = T * (1 - alpha);
            if (test_T < 0.0001f)
            {
                done = true;
                continue;
            }

            int id = collected_id[j];
            for (int ch = 0; ch < CHANNELS; ch++) {
                C[ch] += features[id * CHANNELS + ch] * alpha * T;
            }

            if(get_flag) {
                if (metric_map[pix_id] == 1.0f)
                    atomicAdd(&(metricCount[id]), (int)metric_map[pix_id]); // Cast to int as per prev code
            }
            
            T = test_T;
        }
        last_contributor = contributor;
        contribs++;
    }

    // Write out results
    if (inside) {
        for (int ch = 0; ch < CHANNELS; ch++) {
            // Write to global memory
            out_color[ch * NUM_PIXELS + pix_id] = C[ch]; 
        }
    }
    // All threads that treat valid pixel write out their final
    // rendering data to the frame and auxiliary buffers.
    if (inside)
    {
        final_T[pix_id] = T;
        n_contrib[pix_id] = last_contributor;
        for (int ch = 0; ch < CHANNELS; ch++)
        {
            pixel_colors[ch * HEIGHT * WIDTH + pix_id] = C[ch];
            out_color[ch * HEIGHT * WIDTH + pix_id] = C[ch] + T * bg_color[ch];
        }
    }

    // max reduce the last contributor
    typedef cub::BlockReduce<uint32_t, BLOCK_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_Y> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    last_contributor = BlockReduce(temp_storage).Reduce(last_contributor, cub::Max());
    if (block.thread_rank() == 0) {
        max_contrib[blockIdx.x] = last_contributor;
    }
    t_end = clock64();
    t1 = t_end - t_start;
    // end of 代码段1





    // 代码段2
    t_start = clock64();
    if (get_flag && inside) {
        float actual_diff = 0.0f;
        for (int ch = 0; ch < CHANNELS; ch++) {
            actual_diff += fabsf(out_color[ch * NUM_PIXELS + pix_id] - gt_image[ch * NUM_PIXELS + pix_id]);
        }
        actual_diff /= CHANNELS;

        // Loop over each gaussian 'k' to skip
        for (int k = 0; k < min(BLOCK_SIZE, toDo_initial); k++) {
             // Reset Rendering State for this pass
            float T_new = 1.0f;
            float C_new[CHANNELS] = {0.0f};
            bool done_new = false;

            // Render loop 'j', skipping 'k'
            for (int j = 0; !done_new && j < min(BLOCK_SIZE, toDo_initial); j++) {
                if (j == k) continue; 

                float2 xy = collected_xy[j];
                float2 d = { xy.x - pixf.x, xy.y - pixf.y };
                float4 con_o = collected_conic_opacity[j];
                float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
                
                if (power > 0.0f) continue;
                if (con_o.w < 0) continue;

                float alpha = min(0.99f, con_o.w * expf(power));
                if (alpha < 1.0f / 255.0f) continue;

                float test_T = T_new * (1 - alpha);
                if (test_T < 0.0001f) {
                    done_new = true;
                    continue;
                }
                
                int id = collected_id[j];
                for (int ch = 0; ch < CHANNELS; ch++) {
                    C_new[ch] += features[id * CHANNELS + ch] * alpha * T_new;
                }
                T_new = test_T;
            }

            // Calculate metric for this 'skipped k' scenario
            float current_diff = 0.0f;
            for (int ch = 0; ch < CHANNELS; ch++) {
                current_diff += fabsf(C_new[ch] - gt_image[ch * NUM_PIXELS + pix_id]);
            }
            current_diff /= CHANNELS;
            
            int k_id = collected_id[k];
            atomicAdd(&(metric_per_gs[k_id]), current_diff - actual_diff);
        }
    }
    // end of 代码段2
    t_end = clock64();
    t2 = t_end - t_start;






    // 代码段3
    t_start = clock64();
    if (get_flag && inside) {
        T = 1.0f;
        done = !inside;
        toDo = range.y - range.x;

        float actual_diff = 0.0f;
        for (int ch = 0; ch < CHANNELS; ch++) {
            actual_diff += fabsf(out_color[ch * HEIGHT * WIDTH + pix_id] - gt_image[ch * HEIGHT * WIDTH + pix_id]);
        }
        actual_diff /= CHANNELS;

        // 计算每个删除（添加）每个gs球带来的质量改变
        float out_color_wo_or_w_current_gs = 0.0f;
        float accum_color[CHANNELS] = {0.0f};
        float current_diff = 0.0f;
        float current_color = 0.0f;

        for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
            // Collectively fetch per-Gaussian data from global to shared
            int progress = i * BLOCK_SIZE + block.thread_rank();
            if (range.x + progress < range.y)
            {
                int coll_id = point_list[range.x + progress];
                collected_id[block.thread_rank()] = coll_id;
                collected_xy[block.thread_rank()] = points_xy_image[coll_id];
                collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
                // collected_radius2[block.thread_rank()] = radii[coll_id] * radii[coll_id];
            }
            block.sync();
            
            // Iterate over current batch
            for (int j = 0; inside && j < min(BLOCK_SIZE, toDo); j++) {
				if (done) {
					continue;
				}

                // Resample using conic matrix (cf. "Surface 
                // Splatting" by Zwicker et al., 2001)
                float2 xy = collected_xy[j];
                float2 d = { xy.x - pixf.x, xy.y - pixf.y };
                float4 con_o = collected_conic_opacity[j];
                float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
                if (power > 0.0f) {
					continue;
				}

                if (con_o.w >= 0) {
                    // Eq. (2) from 3D Gaussian splatting paper.
                    // Obtain alpha by multiplying with Gaussian opacity
                    // and its exponential falloff from mean.
                    // Avoid numerical instabilities (see paper appendix). 
                    float alpha = min(0.99f, con_o.w * exp(power));

                    if (alpha < 1.0f / 255.0f) {
                        continue;
					}

                    float test_T = T * (1 - alpha);
                    if (test_T < 0.0001f) {
                        done = true;
                        continue;
                    }

                    // 删除此gs导致质量的变化
                    current_diff = 0.0f;
                    for (int ch = 0; ch < CHANNELS; ch++) {
                        current_color = features[collected_id[j] * CHANNELS + ch] * alpha * T;
						if (alpha < 0.9f) {
							out_color_wo_or_w_current_gs = accum_color[ch] + (out_color[ch * HEIGHT * WIDTH + pix_id] - current_color - accum_color[ch]) / (1 - alpha);
						} else {
							out_color_wo_or_w_current_gs = accum_color[ch];
						}
						out_color_wo_or_w_current_gs = fmaxf(0.0f, fminf(1.0f, out_color_wo_or_w_current_gs));
                        current_diff += fabsf(out_color_wo_or_w_current_gs - gt_image[ch * HEIGHT * WIDTH + pix_id]);

                        // 继续维护正确的累积颜色以便后续计算
                        accum_color[ch] += current_color;
                    }
                    current_diff /= CHANNELS;
					atomicAdd(&(metric_per_gs[collected_id[j]]), current_diff - actual_diff);
					atomicAdd(&(gs_weight[collected_id[j]]), alpha * T);

                    T = test_T;
                }
            }

            block.sync();
        }
    }
    // end of 代码段3
    t_end = clock64();
    t3 = t_end - t_start;

    // Aggregate times
    typedef cub::BlockReduce<long long, BLOCK_SIZE> BlockReduceLong;
    __shared__ typename BlockReduceLong::TempStorage temp_storage_long;

    long long agg_t1 = BlockReduceLong(temp_storage_long).Sum(t1);
    __syncthreads();
    long long agg_t2 = BlockReduceLong(temp_storage_long).Sum(t2);
    __syncthreads();
    long long agg_t3 = BlockReduceLong(temp_storage_long).Sum(t3);

    if (threadIdx.x == 0) {
        atomicAdd((unsigned long long*)&debug_cycles[0], (unsigned long long)agg_t1);
        atomicAdd((unsigned long long*)&debug_cycles[1], (unsigned long long)agg_t2);
        atomicAdd((unsigned long long*)&debug_cycles[2], (unsigned long long)agg_t3);
    }
}

int main() {
    int toDo = 100;
    
    // Allocate device memory
    float *d_features, *d_sampled_T, *d_sampled_ar, *d_metric_map, *d_out_dummy;
    int *d_metricCount;
    float *d_gt_image, *d_out_color, *d_metric_per_gs, *d_gs_weight;
    
    size_t features_size = NUM_GAUSSIANS * CHANNELS * sizeof(float);
    size_t sampled_T_size = BLOCK_SIZE * 10 * sizeof(float);
    size_t sampled_ar_size = BLOCK_SIZE * CHANNELS * 10 * sizeof(float);
    size_t metric_size = NUM_GAUSSIANS * sizeof(int);
    size_t metric_map_size = NUM_PIXELS * sizeof(float);
    size_t image_size = NUM_PIXELS * CHANNELS * sizeof(float);
    size_t gs_metric_size = NUM_GAUSSIANS * sizeof(float);
    
    cudaMalloc(&d_features, features_size);
    cudaMalloc(&d_sampled_T, sampled_T_size);
    cudaMalloc(&d_sampled_ar, sampled_ar_size);
    cudaMalloc(&d_metricCount, metric_size);
    cudaMalloc(&d_metric_map, metric_map_size);
    cudaMalloc(&d_out_dummy, 2 * sizeof(float));
    cudaMalloc(&d_gt_image, image_size);
    cudaMalloc(&d_out_color, image_size);
    cudaMalloc(&d_metric_per_gs, gs_metric_size);
    cudaMalloc(&d_gs_weight, gs_metric_size);
    
    cudaMemset(d_features, 0, features_size);
    cudaMemset(d_metricCount, 0, metric_size);
    cudaMemset(d_gt_image, 0, image_size);
    cudaMemset(d_metric_per_gs, 0, gs_metric_size);
    
    // Additional allocations for kernel 3
    float *d_final_T, *d_pixel_colors, *d_bg_color;
    int *d_n_contrib, *d_max_contrib, *d_point_list;
    float2 *d_points_xy_image;
    float4 *d_conic_opacity;
    long long *d_debug_cycles;
    int num_blocks = 1024;
    
    cudaMalloc(&d_final_T, NUM_PIXELS * sizeof(float));
    cudaMalloc(&d_n_contrib, NUM_PIXELS * sizeof(int));
    cudaMalloc(&d_pixel_colors, image_size);
    cudaMalloc(&d_bg_color, CHANNELS * sizeof(float));
    cudaMalloc(&d_max_contrib, num_blocks * sizeof(int));
    cudaMalloc(&d_point_list, NUM_GAUSSIANS * sizeof(int));
    cudaMalloc(&d_points_xy_image, NUM_GAUSSIANS * sizeof(float2));
    cudaMalloc(&d_conic_opacity, NUM_GAUSSIANS * sizeof(float4));
    cudaMalloc(&d_debug_cycles, 3 * sizeof(long long));
    
    cudaMemset(d_final_T, 0, NUM_PIXELS * sizeof(float));
    cudaMemset(d_debug_cycles, 0, 3 * sizeof(long long));
    
    dim3 grid(num_blocks);
    dim3 block(BLOCK_SIZE);
    int iterations = 100; // Reduce iterations for faster test
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // --- Benchmark Kernel ---
    float milliseconds = 0;
    int2 range = {0, toDo};
    int rounds = (toDo + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Warmup
    benchmark_kernel<<<grid, block>>>(
        toDo, d_features, d_sampled_T, d_sampled_ar, d_metricCount, d_metric_map, d_out_dummy, true, d_gt_image, d_out_color, d_metric_per_gs, d_gs_weight,
        d_final_T, d_n_contrib, d_pixel_colors, d_bg_color, d_max_contrib, d_point_list, d_points_xy_image, d_conic_opacity, range, rounds, d_debug_cycles
    );
    cudaDeviceSynchronize();
    
    cudaMemset(d_debug_cycles, 0, 3 * sizeof(long long));

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        benchmark_kernel<<<grid, block>>>(
            toDo, d_features, d_sampled_T, d_sampled_ar, d_metricCount, d_metric_map, d_out_dummy, true, d_gt_image, d_out_color, d_metric_per_gs, d_gs_weight,
            d_final_T, d_n_contrib, d_pixel_colors, d_bg_color, d_max_contrib, d_point_list, d_points_xy_image, d_conic_opacity, range, rounds, d_debug_cycles
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    long long h_debug_cycles[3];
    cudaMemcpy(h_debug_cycles, d_debug_cycles, 3 * sizeof(long long), cudaMemcpyDeviceToHost);
    
    double total_cycles = (double)h_debug_cycles[0] + (double)h_debug_cycles[1] + (double)h_debug_cycles[2];
    double time1 = milliseconds * (h_debug_cycles[0] / total_cycles);
    double time2 = milliseconds * (h_debug_cycles[1] / total_cycles);
    double time3 = milliseconds * (h_debug_cycles[2] / total_cycles);
    
    printf("Total Time: %f ms\n", milliseconds / iterations);
    printf("Segment 1 Time: %f ms (%.2f%%)\n", time1 / iterations, (h_debug_cycles[0] / total_cycles) * 100.0);
    printf("Segment 2 Time: %f ms (%.2f%%)\n", time2 / iterations, (h_debug_cycles[1] / total_cycles) * 100.0);
    printf("Segment 3 Time: %f ms (%.2f%%)\n", time3 / iterations, (h_debug_cycles[2] / total_cycles) * 100.0);
    
    // Cleanup
    cudaFree(d_features);
    cudaFree(d_sampled_T);
    cudaFree(d_sampled_ar);
    cudaFree(d_metricCount);
    cudaFree(d_metric_map);
    cudaFree(d_out_dummy);
    cudaFree(d_gt_image);
    cudaFree(d_out_color);
    cudaFree(d_metric_per_gs);
    cudaFree(d_gs_weight);
    
    cudaFree(d_final_T);
    cudaFree(d_n_contrib);
    cudaFree(d_pixel_colors);
    cudaFree(d_bg_color);
    cudaFree(d_max_contrib);
    cudaFree(d_point_list);
    cudaFree(d_points_xy_image);
    cudaFree(d_conic_opacity);
    cudaFree(d_debug_cycles);
    
    return 0;
}
