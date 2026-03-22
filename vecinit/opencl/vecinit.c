#include <stdio.h>
#include <stdlib.h>
#include "../ocl_boiler.h"

static cl_event init_vec(cl_command_queue que, cl_kernel init_k, cl_mem vec, int nels) {
    cl_int err;
    cl_event init_evt;
    const size_t gws[1] = { (size_t)nels };

    err = clSetKernelArg(init_k, 0, sizeof(cl_mem), &vec);
    ocl_check(err, "clSetKernelArg");

    err = clEnqueueNDRangeKernel(que, init_k, 1, NULL, gws, NULL, 0, NULL, &init_evt);
    ocl_check(err, "clEnqueueNDRangeKernel");

    return init_evt;
}

static void verify(const int *vec, int nels) {
    for (int i = 0; i < nels; ++i) {
        if (vec[i] != i) {
            fprintf(stderr, "%d != %d @ %d\n", i, vec[i], i);
            exit(4);
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Uso: %s nels\n", argv[0]);
        return 1;
    }

    int nels = atoi(argv[1]);
    if (nels < 1) {
        fprintf(stderr, "nels deve essere almeno 1\n");
        return 2;
    }

    cl_platform_id p = select_platform();
    cl_device_id d = select_device(p);
    cl_context ctx = create_context(p, d);
    cl_command_queue que = create_queue(ctx, d);
    cl_program prog = create_program("vecinit.cl", ctx, d);

    cl_int err;
    const size_t memsize = sizeof(int) * (size_t)nels;

    cl_mem d_vec = clCreateBuffer(
        ctx,
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        memsize,
        NULL,
        &err
    );
    ocl_check(err, "clCreateBuffer");

    cl_kernel init_k = clCreateKernel(prog, "init_k", &err);
    ocl_check(err, "clCreateKernel");

    cl_event init_evt = init_vec(que, init_k, d_vec, nels);

    cl_event read_evt;
    int *h_vec = (int *)clEnqueueMapBuffer(
        que,
        d_vec,
        CL_TRUE,
        CL_MAP_READ,
        0,
        memsize,
        1,
        &init_evt,
        &read_evt,
        &err
    );
    ocl_check(err, "clEnqueueMapBuffer");

    verify(h_vec, nels);

    cl_event unmap_evt;
    err = clEnqueueUnmapMemObject(que, d_vec, h_vec, 0, NULL, &unmap_evt);
    ocl_check(err, "clEnqueueUnmapMemObject");

    clFinish(que);

    printf("init : %.4f ms\n", runtime_ms(init_evt));
    printf("read : %.4f ms\n", runtime_ms(read_evt));
    printf("unmap: %.4f ms\n", runtime_ms(unmap_evt));

    clReleaseEvent(init_evt);
    clReleaseEvent(read_evt);
    clReleaseEvent(unmap_evt);
    clReleaseKernel(init_k);
    clReleaseMemObject(d_vec);
    clReleaseProgram(prog);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);

    return 0;
}