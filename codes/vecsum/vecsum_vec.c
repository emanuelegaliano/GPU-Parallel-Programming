// Implementation of vecsum with vectorization

#include <stdio.h>
#include <stdlib.h>
#include "../ocl_boiler.h"

cl_event init(cl_command_queue que, cl_kernel init_k, cl_mem vec1, cl_mem vec2, cl_int nels, size_t lws_cli)
{
	cl_int err;
	cl_event init_evt;

	const size_t lws[1] = { lws_cli };
	const size_t gws[1] = { round_mul_up(nels, lws[0]) };

	cl_uint arg_index = 0;
	err = clSetKernelArg(init_k, arg_index, sizeof(cl_mem), &vec1);
	ocl_check(err, "init_k set kernel arg %u", arg_index);
    // aggiungo il secondo vettore come argomento del kernel
    ++arg_index;
    err = clSetKernelArg(init_k, arg_index, sizeof(cl_mem), &vec2);
	ocl_check(err, "init_k set kernel arg %u", arg_index);
	++arg_index;
	err = clSetKernelArg(init_k, arg_index, sizeof(nels), &nels);
	ocl_check(err, "init_k set kernel arg %u", arg_index);

	err = clEnqueueNDRangeKernel(que, init_k, 1, NULL, gws, lws, 0, NULL, &init_evt);

	ocl_check(err, "enqueue kernel init_k");
	return init_evt;
}


cl_event sum(cl_command_queue que, cl_kernel sum_k, cl_mem out, cl_mem vec1, cl_mem vec2, cl_int nels, size_t lws_cli, cl_event init_evt)
{
	cl_int err;
	cl_event sum_evt;

	const size_t lws[1] = { lws_cli };
	const size_t gws[1] = { round_mul_up(nels, lws[0]) };

    cl_uint arg_index = 0;
    err = clSetKernelArg(sum_k, arg_index++, sizeof(cl_mem), &out);
    ocl_check(err, "sum_k set arg %u", arg_index);
    err = clSetKernelArg(sum_k, arg_index++, sizeof(cl_mem), &vec1);
    ocl_check(err, "sum_k set arg %u", arg_index);
    err = clSetKernelArg(sum_k, arg_index++, sizeof(cl_mem), &vec2);
    ocl_check(err, "sum_k set arg %u", arg_index);
    err = clSetKernelArg(sum_k, arg_index++, sizeof(nels), &nels);
    ocl_check(err, "sum_k set arg %u", arg_index);

	// la gws indica quanti elementi totali lanciare, la lws come raggrupparli
    err = clEnqueueNDRangeKernel(que, sum_k, 1, NULL, gws, lws, 1, &init_evt, &sum_evt);
    ocl_check(err, "enqueue kernel sum_k");

    return sum_evt;
}


void verify(const cl_int *vec, int nels) {
	for (int i = 0; i < nels; ++i) {
		int expected = nels;
		int computed = vec[i];
		if (expected != computed) {
			fprintf(stderr, "%d != %d @ %d\n", expected, computed, i);
			exit(7);
		}
	}
}


int main(int argc, char *argv[])
{
	if (argc != 4) {
		fprintf(stderr, "%s nels lws vecsize\n", argv[0]);
		exit(1);
	}
	int nels = atoi(argv[1]);
	if (nels < 1) {
		fprintf(stderr, "nels deve essere almeno 1\n");
		exit(2);
	}
	int lws = atoi(argv[2]);
	if (lws < 1) {
		fprintf(stderr, "lws deve essere almeno 1\n");
		exit(3);
	}
	int vecsize = atoi(argv[3]);
	// se (vecsize & (vecsize - 1)) == True allora vecsize non è una potenza di 2
	if ((vecsize < 2) || (vecsize > 16) || (vecsize & (vecsize - 1))) {
		fprintf(stderr, "vecsize deve essere una potenza di 2 compresa tra 2 e 16\n");
		exit(4);
	}
	if(nels & (vecsize - 1)){
		fprintf(stderr, "nels %d non è multiplo di vecsize %d\n", nels, vecsize);
		exit(5);
	}

	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("vecsum_vec.cl", ctx, d);

	const size_t memsize = sizeof(int)*nels;

    cl_int err;
	cl_mem d_vec1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, memsize, NULL, &err);
	ocl_check(err, "clCreateBuffer d_vec1 fallito");
    cl_mem d_vec2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, memsize, NULL, &err);
	ocl_check(err, "clCreateBuffer d_vec2 fallito");
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, memsize, NULL, &err);
	ocl_check(err, "clCreateBuffer d_out fallito");

	cl_kernel init_k = clCreateKernel(prog, "init_k", &err);
	ocl_check(err, "clCreateKernel init_k fallito");
	char sum_k_name[11] = {0}; // ci assicuriamo che ci sia il NULL byte
	snprintf(sum_k_name, 10, "sum_v%d_k", vecsize);
	cl_kernel sum_k = clCreateKernel(prog, sum_k_name, &err);
	ocl_check(err, "clCreateKernel %s fallito", sum_k_name);

	cl_event init_evt = init(que, init_k, d_vec1, d_vec2, nels, lws);
	
	// nels/vecsize -> si passa il numero di vettori
	cl_event sum_evt = sum(que, sum_k, d_out, d_vec1, d_vec2, nels/vecsize, lws, init_evt);

	cl_event wait_list[1] = { sum_evt };

	cl_event map_evt;
	cl_int * h_vec = clEnqueueMapBuffer(que, d_out, CL_TRUE, CL_MAP_READ, 0, memsize, 1, wait_list, &map_evt, &err);
	ocl_check(err, "map buffer d_out");

	verify(h_vec, nels);

	cl_event unmap_evt;
	err = clEnqueueUnmapMemObject(que, d_out, h_vec, 0, NULL, &unmap_evt);
	ocl_check(err, "unmap buffer d_out");

	clFinish(que);

	cl_ulong init_ns = runtime_ns(init_evt);
    cl_ulong sum_ns = runtime_ns(sum_evt);
	cl_ulong map_ns = runtime_ns(map_evt);
	cl_ulong unmap_ns = runtime_ns(unmap_evt);
	printf("init: %.4gms %.4gGB/s\n", init_ns*1.0e-6, 2*memsize/(double)init_ns);
    // Il calcolo della banda passante non cambia (il numero di elementi processati è lo stesso)
    printf("sum: %.4gms %.4gGB/s\n", sum_ns*1.0e-6, 3*memsize/(double)sum_ns);
	printf("map: %.4gms %.4gGB/s\n", map_ns*1.0e-6, memsize/(double)map_ns);
	printf("unmap: %.4gms %.4gGB/s\n", unmap_ns*1.0e-6, memsize/(double)unmap_ns);

	/* Cleanup */
	clReleaseEvent(init_evt);
	clReleaseEvent(sum_evt);
	clReleaseEvent(map_evt);
	clReleaseEvent(unmap_evt);
	clReleaseKernel(init_k);
	clReleaseKernel(sum_k);
	clReleaseProgram(prog);
	clReleaseMemObject(d_vec1);
	clReleaseMemObject(d_vec2);
	clReleaseMemObject(d_out);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);

	return 0;
}