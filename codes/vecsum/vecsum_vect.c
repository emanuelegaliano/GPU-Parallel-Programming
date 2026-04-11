#include <stdio.h>
#include <stdlib.h>
#include "../ocl_boiler.h"
#include "../ocl_bench.h"
#include <CL/cl.h>

#define SUM_KERNEL_NAME "sum_k"

int parse_args(int argc, char** argv, size_t* n);
int select_gws(size_t n, size_t* gws, size_t* lws);
void init_setup(const cl_kernel init_k, cl_mem vec1, cl_mem vec2, size_t n);
cl_event sum_setup(cl_kernel sum_k, cl_command_queue queue,
                   cl_mem out, cl_mem vec1, cl_mem vec2,
                   cl_int nels, const size_t* gws, const size_t* lws);
void verify(const cl_int* result, size_t n);

void bench_kernel(cl_kernel k, cl_command_queue queue,
                  cl_mem out, cl_mem vec1, cl_mem vec2,
                  cl_int nels, size_t gws_count,
                  const size_t* lws, size_t bench_bytes,
                  const char* label)
{
    size_t gws[1] = { gws_count };
    cl_event e = sum_setup(k, queue, out, vec1, vec2, nels, gws, lws);
    clFinish(queue);
    ocl_bench_print(label, ocl_bench_event(e, bench_bytes));
}

int main(int argc, char** argv) {
    puts("Starting vecsum_vect...");
    size_t n;
    if (parse_args(argc, argv, &n)) {
        return 1;
    }

    // ==== OPENCL BOILERPLATE ======
    puts("Initializing OpenCL...");
    cl_platform_id platform = select_platform();
    cl_device_id device = select_device(platform);
    cl_context context = create_context(platform, device);
    cl_command_queue queue = create_queue(context, device);
    cl_program program = create_program("vecsum_vect.cl", context, device);

    // ====== CREATE BUFFERS ======
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
        context, 
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
        memsize, 
        NULL, 
        &err
    );
    ocl_check(err, "create vec1");

    cl_mem vec2 = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
        memsize, 
        NULL, 
        &err
    );
    ocl_check(err, "create vec2");

    cl_mem out = clCreateBuffer(
        context, 
        CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, 
        memsize, 
        NULL, 
        &err
    );
    ocl_check(err, "create out");

    // ====== CREATE KERNELS ======
    puts("Creating kernels...");
    const char* init_k_name = "init_k";
    const char* sum_k_name = SUM_KERNEL_NAME;
    
    cl_kernel init_k = clCreateKernel(program, init_k_name, &err);
    ocl_check(err, "create init_kernel");

    cl_kernel sum_k = clCreateKernel(program, sum_k_name, &err);
    ocl_check(err, "create sum_kernel");

    cl_kernel sum_s2_k      = clCreateKernel(program, "sum_s2_k",      &err); ocl_check(err, "create sum_s2_k");
    cl_kernel sum_s4_k      = clCreateKernel(program, "sum_s4_k",      &err); ocl_check(err, "create sum_s4_k");
    cl_kernel sum_s2s_k     = clCreateKernel(program, "sum_s2s_k",     &err); ocl_check(err, "create sum_s2s_k");
    cl_kernel sum_s4s_k     = clCreateKernel(program, "sum_s4s_k",     &err); ocl_check(err, "create sum_s4s_k");
    cl_kernel sum_s4sx_k    = clCreateKernel(program, "sum_s4sx_k",    &err); ocl_check(err, "create sum_s4sx_k");
    cl_kernel sum_v2_k      = clCreateKernel(program, "sum_v2_k",      &err); ocl_check(err, "create sum_v2_k");
    cl_kernel sum_v4_k      = clCreateKernel(program, "sum_v4_k",      &err); ocl_check(err, "create sum_v4_k");
    cl_kernel sum_v8_k      = clCreateKernel(program, "sum_v8_k",      &err); ocl_check(err, "create sum_v8_k");
    cl_kernel sum_v16_k     = clCreateKernel(program, "sum_v16_k",     &err); ocl_check(err, "create sum_v16_k");
    cl_kernel sum_sliding_k = clCreateKernel(program, "sum_sliding_k", &err); ocl_check(err, "create sum_sliding_k");

    puts("Setting kernel args...");
    size_t lws[1] = {256};
    size_t gws[1];
    if (select_gws(n, gws, lws)) {
        return 1;
    }

    printf("Global work size: %zu\n", gws[0]);
    cl_event init_evt, sum_evt;

    // Init kernels setup and args
    init_setup(init_k, vec1, vec2, n);
    err = clEnqueueNDRangeKernel(queue, init_k, 1, NULL, gws, lws, 0, NULL, &init_evt);
    ocl_check(err, "enqueue init_k");

    sum_evt = sum_setup(sum_k, queue, out, vec1, vec2, (cl_int)n, gws, lws);

    puts("Waiting for kernels to finish...");
    cl_event map_evt;
    const cl_event map_wait[1] = { sum_evt };

    cl_int* result = clEnqueueMapBuffer(
        queue, 
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
    ocl_check(err, "map output buffer");

    // ====== VERIFY RESULTS ======
    puts("Verifying results...");
    verify(result, n);
    puts("Verification successful!");
    
    // UNMAP OUTPUT BUFFER
    puts("Unmapping output buffer...");
    err = clEnqueueUnmapMemObject(queue, out, result, 0, NULL, NULL);
    ocl_check(err, "unmap output buffer");

    clFinish(queue);
    

    // ====== BENCHMARK ======
    puts("Starting benchmark...");
    const size_t bench_bytes = ocl_bench_bytes(2, 1, n * sizeof(cl_int));

    bench_kernel(sum_k,         queue, out, vec1, vec2, (cl_int)n,     round_mul_up(n,    lws[0]), lws, bench_bytes, "sum_k");
    bench_kernel(sum_s2_k,      queue, out, vec1, vec2, (cl_int)n,     round_mul_up(n/2,  lws[0]), lws, bench_bytes, "sum_s2_k");
    bench_kernel(sum_s4_k,      queue, out, vec1, vec2, (cl_int)n,     round_mul_up(n/4,  lws[0]), lws, bench_bytes, "sum_s4_k");
    bench_kernel(sum_s2s_k,     queue, out, vec1, vec2, (cl_int)n,     round_mul_up(n/2,  lws[0]), lws, bench_bytes, "sum_s2s_k");
    bench_kernel(sum_s4s_k,     queue, out, vec1, vec2, (cl_int)n,     round_mul_up(n/4,  lws[0]), lws, bench_bytes, "sum_s4s_k");
    bench_kernel(sum_s4sx_k,    queue, out, vec1, vec2, (cl_int)n,     round_mul_up(n/4,  lws[0]), lws, bench_bytes, "sum_s4sx_k");
    bench_kernel(sum_v2_k,      queue, out, vec1, vec2, (cl_int)(n/2), round_mul_up(n/2,  lws[0]), lws, bench_bytes, "sum_v2_k");
    bench_kernel(sum_v4_k,      queue, out, vec1, vec2, (cl_int)(n/4), round_mul_up(n/4,  lws[0]), lws, bench_bytes, "sum_v4_k");
    bench_kernel(sum_v8_k,      queue, out, vec1, vec2, (cl_int)(n/8), round_mul_up(n/8,  lws[0]), lws, bench_bytes, "sum_v8_k");
    bench_kernel(sum_v16_k,     queue, out, vec1, vec2, (cl_int)(n/16),round_mul_up(n/16, lws[0]), lws, bench_bytes, "sum_v16_k");
    bench_kernel(sum_sliding_k, queue, out, vec1, vec2, (cl_int)n,     round_mul_up(n,    lws[0]), lws, bench_bytes, "sum_sliding_k");

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

cl_event sum_setup(cl_kernel sum_k, cl_command_queue queue,
                   cl_mem out, cl_mem vec1, cl_mem vec2,
                   cl_int nels, const size_t* gws, const size_t* lws)
{
    cl_int err;
    cl_uint arg = 0;
    cl_event evt;

    err = clSetKernelArg(sum_k, arg, sizeof(cl_mem), &out);
    ocl_check(err, "set sum_k arg 0 (out)");   arg++;

    err = clSetKernelArg(sum_k, arg, sizeof(cl_mem), &vec1);
    ocl_check(err, "set sum_k arg 1 (vec1)");  arg++;

    err = clSetKernelArg(sum_k, arg, sizeof(cl_mem), &vec2);
    ocl_check(err, "set sum_k arg 2 (vec2)");  arg++;

    err = clSetKernelArg(sum_k, arg, sizeof(cl_int), &nels);
    ocl_check(err, "set sum_k arg 3 (nels)");

    err = clEnqueueNDRangeKernel(queue, sum_k, 1,
                                 NULL, gws, lws,
                                 0, NULL, &evt);
    ocl_check(err, "enqueue sum_k");

    return evt;
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