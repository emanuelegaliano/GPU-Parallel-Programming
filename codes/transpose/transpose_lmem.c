/* Host code of transpose with local memory usage */
#include <stdio.h>
#include <stdlib.h>
#include "../ocl_boiler.h"

cl_event init(cl_command_queue que, cl_kernel init_k, cl_mem mat, cl_int nrows, cl_int ncols,
	cl_int lws_cli)
{
	cl_int err;
	cl_event init_evt;

	const size_t lws[2] = { lws_cli, lws_cli };
	const size_t gws[2] = { round_mul_up(ncols, lws[0]), round_mul_up(nrows, lws[1]) };

	cl_uint arg_index = 0;
	err = clSetKernelArg(init_k, arg_index, sizeof(cl_mem), &mat);
	ocl_check(err, "init_k set kernel arg %u", arg_index);
	++arg_index;
	err = clSetKernelArg(init_k, arg_index, sizeof(nrows), &nrows);
	ocl_check(err, "init_k set kernel arg %u", arg_index);
	++arg_index;
	err = clSetKernelArg(init_k, arg_index, sizeof(ncols), &ncols);
	ocl_check(err, "init_k set kernel arg %u", arg_index);

	err = clEnqueueNDRangeKernel(que, init_k, 2, NULL, gws, lws, 0, NULL, &init_evt);

	ocl_check(err, "enqueue kernel init_k");
	return init_evt;
}

cl_event transpose(cl_command_queue que, cl_kernel transpose_k,
	cl_mem tras, cl_mem orig, cl_int t_rows, cl_int t_cols,
	cl_int lws_cli,	cl_event init_evt)
{
	cl_int err;
	cl_event transpose_evt;

	const size_t lws[2] = { lws_cli, lws_cli };
	const size_t gws[2] = { round_mul_up(t_rows, lws[0]), round_mul_up(t_cols, lws[1]) };
	// printf("GWS: %zu x %zu\n", gws[0], gws[1]);

	cl_uint arg_index = 0;
	err = clSetKernelArg(transpose_k, arg_index, sizeof(cl_mem), &tras);
	ocl_check(err, "transpose_k set kernel arg %u", arg_index);
	++arg_index;
	err = clSetKernelArg(transpose_k, arg_index, sizeof(cl_mem), &orig);
	ocl_check(err, "transpose_k set kernel arg %u", arg_index);
	++arg_index;
	err = clSetKernelArg(transpose_k, arg_index, sizeof(t_rows), &t_rows);
	ocl_check(err, "transpose_k set kernel arg %u", arg_index);
	++arg_index;
	err = clSetKernelArg(transpose_k, arg_index, sizeof(t_cols), &t_cols);
	ocl_check(err, "transpose_k set kernel arg %u", arg_index);
	++arg_index;
	err = clSetKernelArg(transpose_k, arg_index, lws_cli*lws_cli*sizeof(cl_int), NULL);
	ocl_check(err, "transpose_k set kernel arg %u", arg_index);

	err = clEnqueueNDRangeKernel(que, transpose_k, 2, NULL, gws, lws, 1, &init_evt, &transpose_evt);

	ocl_check(err, "enqueue kernel transpose_k");
	return transpose_evt;
}

void verify(const cl_int *mat, int nrows, int ncols) {
	for (int r = 0; r < nrows; ++r) {
		for (int c = 0; c < ncols; ++c) {
			int expected = c - r;
			int computed = mat[r*ncols+c];
			if (expected != computed) {
				fprintf(stderr, "%d != %d @ (%d, %d)\n",
					expected, computed, r, c);
				exit(7);
			}
		}
	}
}


int main(int argc, char *argv[])
{
	if (argc != 4) {
		fprintf(stderr, "%s nrows ncols lws\n", argv[0]);
		exit(1);
	}
	int nrows_orig = atoi(argv[1]);
	if (nrows_orig < 1) {
		fprintf(stderr, "nrows deve essere almeno 1\n");
		exit(2);
	}
	int ncols_orig = atoi(argv[2]);
	if (ncols_orig < 1) {
		fprintf(stderr, "ncols deve essere almeno 1\n");
		exit(3);
	}
	int lws = atoi(argv[3]);
	if (lws < 1) {
		fprintf(stderr, "lws deve essere almeno 1\n");
		exit(4);
	}

	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("transpose.ocl", ctx, d);

	const size_t memsize = sizeof(int)*nrows_orig*ncols_orig;

	cl_int err;
	cl_mem d_orig = clCreateBuffer(ctx, CL_MEM_READ_WRITE, memsize, NULL, &err);
	ocl_check(err, "clCreateBuffer fallito (orig)");
	cl_mem d_tras = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, memsize, NULL, &err);
	ocl_check(err, "clCreateBuffer fallito (tras)");

	const char *kname_init = "init_k";
	const char *kname_tras = "transpose_lmem_k";
	cl_kernel init_k = clCreateKernel(prog, kname_init, &err);
	ocl_check(err, "clCreateKernel %s fallito", kname_init);
	cl_kernel transpose_k = clCreateKernel(prog, kname_tras, &err);
	ocl_check(err, "clCreateKernel %s fallito", kname_tras);

	cl_event init_evt = init(que, init_k, d_orig, nrows_orig, ncols_orig, lws);

	cl_int nrows_tras = ncols_orig;
	cl_int ncols_tras = nrows_orig;
	cl_event transpose_evt = transpose(que, transpose_k, d_tras, d_orig, nrows_tras, ncols_tras, lws, init_evt);

	cl_event wait_list[] = { init_evt, transpose_evt };
	cl_event read_evt;

	cl_int * h_tras = clEnqueueMapBuffer(que, d_tras, CL_TRUE, CL_MAP_READ, 0, memsize, 2, wait_list, &read_evt, &err);
	ocl_check(err, "read buffer d_tras");

	verify(h_tras, nrows_tras, ncols_tras);

	cl_event unmap_evt;
	err = clEnqueueUnmapMemObject(que, d_tras, h_tras, 0, NULL, &unmap_evt);
	ocl_check(err, "unmap buffer d_tras");

	clFinish(que);

	cl_ulong init_ns = runtime_ns(init_evt);
	cl_ulong transpose_ns = runtime_ns(transpose_evt);
	cl_ulong read_ns = runtime_ns(read_evt);
	cl_ulong unmap_ns = runtime_ns(unmap_evt);
	printf("init: %.4gms %.4gGB/s\n", init_ns*1.0e-6, memsize/(double)init_ns);
	printf("transpose: %.4gms %.4gGB/s\n", transpose_ns*1.0e-6, 2*memsize/(double)transpose_ns);
	printf("read: %.4gms %.4gGB/s\n", read_ns*1.0e-6, memsize/(double)read_ns);
	printf("unmap: %.4gms %.4gGB/s\n", unmap_ns*1.0e-6, memsize/(double)unmap_ns);

    /* Cleanup */
	clReleaseEvent(init_evt);
    clReleaseEvent(transpose_evt);
    clReleaseEvent(read_evt);
    clReleaseEvent(unmap_evt);
    clReleaseKernel(init_k);
    clReleaseKernel(transpose_k);
    clReleaseProgram(prog);
    clReleaseMemObject(d_orig);
    clReleaseMemObject(d_tras);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);

	return 0;
}

