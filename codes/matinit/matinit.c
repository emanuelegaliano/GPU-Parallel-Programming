#include <stdio.h>
#include <stdlib.h>
#include "../ocl_boiler.h"

cl_event init(cl_command_queue que, cl_kernel init_k, cl_mem mat, cl_int nrows, cl_int ncols, cl_int lws_cli)
{
	cl_int err;
	cl_event init_evt;

	// local work size = quadrato di dimensioni lws_cli*lws_cli
	const size_t lws[2] = { lws_cli, lws_cli };
	// griglia di lancio -> il 1° indice è di riga, il 2° è di colonna
	const size_t gws[2] = { round_mul_up(ncols, lws[0]), round_mul_up(nrows, lws[1]) };
	printf("GWS: %zu x %zu\n", gws[0], gws[1]);

	cl_uint arg_index = 0;
	err = clSetKernelArg(init_k, arg_index, sizeof(cl_mem), &mat);
	ocl_check(err, "init_k set kernel arg %u", arg_index);
	++arg_index;
	err = clSetKernelArg(init_k, arg_index, sizeof(nrows), &nrows);
	ocl_check(err, "init_k set kernel arg %u", arg_index);
	++arg_index;
	err = clSetKernelArg(init_k, arg_index, sizeof(ncols), &ncols);
	ocl_check(err, "init_k set kernel arg %u", arg_index);

	// 2 -> griglia di lancio 2D
	err = clEnqueueNDRangeKernel(que, init_k, 2, NULL, gws, lws, 0, NULL, &init_evt);

	ocl_check(err, "enqueue kernel init_k");
	return init_evt;
}


void verify(const cl_int *mat, int nrows, int ncols) {
	for (int r = 0; r < nrows; ++r) {
		for (int c = 0; c < ncols; ++c) {
			int expected = r-c;
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
	int nrows = atoi(argv[1]);
	if (nrows < 1) {
		fprintf(stderr, "nrows deve essere almeno 1\n");
		exit(2);
	}
	int ncols = atoi(argv[2]);
	if (ncols < 1) {
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
	cl_program prog = create_program("matinit.ocl", ctx, d);

	const size_t memsize = sizeof(int)*nrows*ncols;

	cl_int err;
	cl_mem d_vec = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, memsize, NULL, &err);
	ocl_check(err, "clCreateBuffer fallito");

	const char *kname = "init_k";
	cl_kernel init_k = clCreateKernel(prog, kname, &err);
	ocl_check(err, "clCreateKernel %s fallito", kname);

	cl_event init_evt = init(que, init_k, d_vec, nrows, ncols, lws);

	cl_event wait_list[1] = { init_evt };
	cl_event read_evt;

	cl_int *h_vec = clEnqueueMapBuffer(que, d_vec, CL_TRUE, CL_MAP_READ, 0, memsize, 1, wait_list, &read_evt, &err);
	ocl_check(err, "read buffer d_vec");

	verify(h_vec, nrows, ncols);

	cl_event unmap_evt;
	err = clEnqueueUnmapMemObject(que, d_vec, h_vec, 0, NULL, &unmap_evt);
	ocl_check(err, "unmap buffer d_vec");

	clFinish(que);

	cl_ulong init_ns = runtime_ns(init_evt);
	cl_ulong read_ns = runtime_ns(read_evt);
	cl_ulong unmap_ns = runtime_ns(unmap_evt);
	printf("init: %.4gms %.4gGB/s\n", init_ns*1.0e-6, memsize/(double)init_ns);
	printf("read: %.4gms %.4gGB/s\n", read_ns*1.0e-6, memsize/(double)read_ns);
	printf("unmap: %.4gms %.4gGB/s\n", unmap_ns*1.0e-6, memsize/(double)unmap_ns);

	/* Cleanup */
	clReleaseEvent(init_evt);
	clReleaseEvent(read_evt);
	clReleaseEvent(unmap_evt);
	clReleaseKernel(init_k);
	clReleaseProgram(prog);
	clReleaseMemObject(d_vec);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);

	return 0;
}