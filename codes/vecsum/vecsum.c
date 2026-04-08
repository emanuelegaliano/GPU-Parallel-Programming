#include <stdio.h>
#include <stdlib.h>
#include "../ocl_boiler.h"
#include <CL/cl.h>



int parse_args(int argc, char** argv, size_t* n);
int select_gws(size_t n, size_t* gws, size_t* lws);
void init_setup(const cl_kernel init_k, cl_mem vec1, cl_mem vec2, size_t n);
void sum_setup(const cl_kernel sum_k, cl_mem vec1, cl_mem vec2, cl_mem result, size_t n);
void verify(const cl_int* result, size_t n);

int main(int argc, char** argv) {

    // ====== PARSE ARGS ======= 
    size_t n;
    if (parse_args(argc, argv, &n)) {
        return 1;
    }

    printf("Vecsum: n = %zu\n", n);
    puts("Initializing OpenCL...");

    // ====== OPENCL BOILERPLATE =======
    cl_platform_id p = select_platform();
    cl_device_id d = select_device(p);

    cl_context ctx = create_context(p, d);
    cl_command_queue que = create_queue(ctx, d);
    cl_program prog = create_program("vecsum.cl", ctx, d);

    // ====== CREATE VECTORS ======
    puts("Creating buffers...");
    const size_t memsize = sizeof(int)*n;
    cl_int err;

    /** Using:
     * CL_MEM_WRITE_ONLY: the kernel will only write to this buffer, 
     *  so the OpenCL implementation can optimize for that
     * CL_MEM_ALLOC_HOST_PTR: the buffer will be allocated in a way 
     *  that allows the host to map it (see clEnqueueMapBuffer) 
     *  and access it directly, without copying it to host memory. 
     *  This is useful for the output buffer, since we will need to 
     *  read the results back to the host after the kernel finishes.
     */
    
    cl_mem vec1 = clCreateBuffer(
        ctx, 
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
        memsize, 
        NULL, 
        &err
    );
    ocl_check(err, "create vec1");

    cl_mem vec2 = clCreateBuffer(
        ctx, 
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
        memsize, 
        NULL, 
        &err
    );
    ocl_check(err, "create vec2");

    cl_mem out = clCreateBuffer(
        ctx, 
        CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, 
        memsize, 
        NULL, 
        &err
    );
    ocl_check(err, "create out");

    // ====== CREATE KERNELS ======
    puts("Creating kernels...");
    const char* init_k_name = "init_k";
    const char* sum_k_name = "sum_k";

    cl_kernel init_k = clCreateKernel(prog, init_k_name, &err);
    ocl_check(err, "create init_kernel");

    cl_kernel sum_k = clCreateKernel(prog, sum_k_name, &err);
    ocl_check(err, "create sum_kernel");

    // ====== SET KERNEL ARGS ======
    puts("Setting kernel args...");
    size_t lws[1] = {256};
    size_t gws[1];
    if (select_gws(n, gws, lws)) {
        return 1;
    }

    // Init kernels setup and args
    printf("Global work size: %zu\n", gws[0]);
    cl_event init_evt, sum_evt, wait_list[2];

    init_setup(init_k, vec1, vec2, n);
    err = clEnqueueNDRangeKernel(que, init_k, 1, NULL, gws, lws, 0, NULL, &init_evt);
    ocl_check(err, "enqueue init_k");
    wait_list[0] = init_evt;
    
    sum_setup(sum_k, vec1, vec2, out, n);
    err = clEnqueueNDRangeKernel(que, sum_k, 1, NULL, gws, lws, 1, wait_list, &sum_evt);
    ocl_check(err, "enqueue sum_k");
    wait_list[1] = sum_evt;

    // ====== WAIT FOR KERNELS TO FINISH ======
    puts("Waiting for kernels to finish...");
    cl_event map_evt;
    const cl_event map_wait[1] = { sum_evt };

    // ====== MAP OUTPUT BUFFER AND VERIFY RESULTS ======
    cl_int* result = clEnqueueMapBuffer(
        que, 
        out, 
        CL_TRUE, 
        CL_MAP_READ, 
        0, 
        memsize, 
        1, 
        map_wait, 
        &map_evt, 
        &err
    );
    ocl_check(err, "map out buffer");

    verify(result, n);
    puts("Verification passed!");

    // ======= UNMAP OUTPUT BUFFER =======
    puts("Unmapping output buffer...");
    err = clEnqueueUnmapMemObject(que, out, result, 0, NULL, NULL);
    ocl_check(err, "unmap out buffer");

    clFinish(que);

    // ====== PRINT TIMINGS ======
    double init_time = runtime_ms(init_evt);
    double sum_time = runtime_ms(sum_evt);
    double total_time = total_runtime_ms(init_evt, sum_evt);

    printf("Init kernel time: %.3f ms\n", init_time);
    printf("Init kernel bandwidth: %.3f GB/s\n", 2.0*memsize/(init_time*1.0e6));
    printf("Sum kernel time: %.3f ms\n", sum_time);
    printf("Sum kernel bandwidth: %.3f GB/s\n", 3.0*memsize/(sum_time*1.0e6));
    printf("Total time (init + sum): %.3f ms\n", total_time);

    // ====== CLEANUP ======
    puts("Cleaning up...");
    clReleaseMemObject(vec1);
    clReleaseMemObject(vec2);
    clReleaseMemObject(out);
    clReleaseKernel(init_k);
    clReleaseKernel(sum_k);
    clReleaseProgram(prog);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);

    return 0;
}

void verify(const cl_int* result, size_t n) {
    for(size_t i = 0; i < n; i++) {
        int expected = n;
        int computed = result[i];
        if (computed != expected) {
            fprintf(stderr, "Verification failed at index %zu: expected %d, got %d\n", i, expected, computed);
            exit(1);
        }
    }
}

void sum_setup(const cl_kernel sum_k, cl_mem vec1, cl_mem vec2, cl_mem result, size_t n) {
    cl_int err;
    cl_uint arg_index = 0;

    err = clSetKernelArg(sum_k, arg_index, sizeof(cl_mem), &result);
    ocl_check(err, "set sum_k arg 0");
    arg_index++;

    err = clSetKernelArg(sum_k, arg_index, sizeof(cl_mem), &vec1);
    ocl_check(err, "set sum_k arg 1");
    arg_index++;

    err = clSetKernelArg(sum_k, arg_index, sizeof(cl_mem), &vec2);
    ocl_check(err, "set sum_k arg 2");
    arg_index++;
    
    cl_int n_int = (cl_int)n;
    err = clSetKernelArg(sum_k, arg_index, sizeof(cl_int), &n_int);
    ocl_check(err, "set sum_k arg 3");
}

void init_setup(const cl_kernel init_k, cl_mem vec1, cl_mem vec2, size_t n) {
    cl_int err;
    cl_uint arg_index = 0;

    err = clSetKernelArg(init_k, arg_index, sizeof(cl_mem), &vec1);
    ocl_check(err, "set init_k arg 0");
    arg_index++;

    err = clSetKernelArg(init_k, arg_index, sizeof(cl_mem), &vec2);
    ocl_check(err, "set init_k arg 1");
    arg_index++;
    
    cl_int n_int = (cl_int)n;
    err = clSetKernelArg(init_k, arg_index, sizeof(cl_int), &n_int);
    ocl_check(err, "set init_k arg 2");
}

int select_gws(size_t n, size_t* gws, size_t* lws) {
    if (n == 0) {
        fprintf(stderr, "n must be > 0\n");
        return 1;
    }
    if (lws == NULL || gws == NULL) {
        fprintf(stderr, "lws and gws must be non-NULL\n");
        return 1;
    }
    if (lws[0] == 0) {
        fprintf(stderr, "lws[0] must be > 0\n");
        return 1;
    }

    gws[0] = round_mul_up(n, lws[0]);
    return 0;
}

int parse_args(int argc, char** argv, size_t* n) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <n>\n", argv[0]);
        return 1;
    }
    *n = atoi(argv[1]);
    return 0;
}

