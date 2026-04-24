/* Host vecsmooth code with vectorization */

#include <stdio.h>
#include <stdlib.h>

#include "../ocl_boiler.h"

cl_event init(cl_command_queue que, cl_kernel init_k, cl_mem vec, cl_int nels,
	size_t lws_cli)
{
	cl_int err;
	cl_event init_evt;

	const size_t lws[1] = { lws_cli };
	const size_t gws[1] = { round_mul_up(nels, lws[0]) };

	cl_uint arg_index = 0;
	err = clSetKernelArg(init_k, arg_index, sizeof(cl_mem), &vec);
	ocl_check(err, "init_k set kernel arg %u", arg_index);
	++arg_index;
	err = clSetKernelArg(init_k, arg_index, sizeof(nels), &nels);
	ocl_check(err, "init_k set kernel arg %u", arg_index);

	err = clEnqueueNDRangeKernel(que, init_k, 1, NULL, gws, lws, 0, NULL, &init_evt);

	ocl_check(err, "enqueue kernel init_k");
	return init_evt;
}

cl_event smooth(cl_command_queue que, cl_kernel smooth_k,
	cl_mem out, cl_mem in, cl_int nels,
	size_t lws_cli, cl_event init_evt)
{
	cl_int err;
	cl_event smooth_evt;

	const size_t lws[1] = { lws_cli };
	const size_t gws[1] = { round_mul_up(nels, lws[0]) };

	cl_uint arg_index = 0;
	err = clSetKernelArg(smooth_k, arg_index, sizeof(cl_mem), &out);
	ocl_check(err, "smooth_k set kernel arg %u", arg_index);
	++arg_index;
	err = clSetKernelArg(smooth_k, arg_index, sizeof(cl_mem), &in);
	ocl_check(err, "smooth_k set kernel arg %u", arg_index);
	++arg_index;
	err = clSetKernelArg(smooth_k, arg_index, sizeof(nels), &nels);
	ocl_check(err, "smooth_k set kernel arg %u", arg_index);

	err = clEnqueueNDRangeKernel(que, smooth_k, 1, NULL, gws, lws, 1, &init_evt, &smooth_evt);

	ocl_check(err, "enqueue kernel smooth_k");
	return smooth_evt;
}


void verify(const cl_int *vec, int nels) {
	for (int i = 0; i < nels; ++i) {
		int expected = i < nels - 1 ? i : i - 1;
		int computed = vec[i];
		if (expected != computed) {
			fprintf(stderr, "%d != %d @ %d\n",
				expected, computed, i);
			exit(7);
		}
	}
}


int main(int argc, char *argv[])
{
	if (argc != 3) {
		fprintf(stderr, "%s nels lws\n", argv[0]);
		exit(1);
	}
	int nels = atoi(argv[1]);
	if (nels < 1) {
		fprintf(stderr, "nels deve essere almeno 1\n");
		exit(2);
	}
	if (nels & 3) {
		fprintf(stderr, "nels deve essere multiplo di 4\n");
		exit(2);
	}
	int lws = atoi(argv[2]);
	if (lws < 1) {
		fprintf(stderr, "lws deve essere almeno 1\n");
		exit(2);
	}

	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);

	cl_program prog = create_program("vecsmooth.ocl", ctx, d);

	const size_t memsize = sizeof(int)*nels;

	cl_int err;
	cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, memsize, NULL, &err);
	ocl_check(err, "clCreateBuffer d_vec1 fallito");
	cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, memsize, NULL, &err);
	ocl_check(err, "clCreateBuffer d_out fallito");

	cl_kernel init_k = clCreateKernel(prog, "init_k", &err);
	ocl_check(err, "clCreateKernel init_k fallito");
    // usiamo il kernel che usa i tipi vettoriali (int 4)
	cl_kernel smooth_k = clCreateKernel(prog, "smooth_v4_k", &err);
	ocl_check(err, "clCreateKernel smooth_k fallito");

	cl_event init_evt = init(que, init_k, d_in, nels, lws);

	cl_event smooth_evt = smooth(que, smooth_k, d_out, d_in, nels/4, lws, init_evt);

	cl_event wait_list[1] = { smooth_evt };
	cl_event map_evt;
	cl_int * h_vec = clEnqueueMapBuffer(que, d_out, CL_TRUE, CL_MAP_READ, 0, memsize, 1, wait_list, &map_evt, &err);
	ocl_check(err, "map buffer d_out");

	verify(h_vec, nels);

	cl_event unmap_evt;
	err = clEnqueueUnmapMemObject(que, d_out, h_vec, 0, NULL, &unmap_evt);
	ocl_check(err, "unmap buffer d_out");

	clFinish(que);

	cl_ulong init_ns = runtime_ns(init_evt);
	cl_ulong smooth_ns = runtime_ns(smooth_evt);
	cl_ulong map_ns = runtime_ns(map_evt);
	printf("init: %.4gms %.4gGB/s\n", init_ns*1.0e-6, memsize/(double)init_ns);
    // banda passante -> 1 + (vecsize+2)/vecsize
	printf("smooth: %.4gms %.4gGB/s, %.4gGE/s\n", smooth_ns*1.0e-6, (1 + 6.0/4)*memsize/(double)smooth_ns, nels/(double)smooth_ns);
	printf("map: %.4gms %.4gGB/s\n", map_ns*1.0e-6, memsize/(double)map_ns);

	return 0;
}

