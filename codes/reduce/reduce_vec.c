#include <stdio.h>
#include <stdlib.h>

#include "ocl_boiler.h"

cl_event init(cl_command_queue que, cl_kernel init_k, cl_mem vec1, cl_int nels,
	size_t lws_cli)
{
	cl_int err;
	cl_event init_evt;

	const size_t lws[1] = { lws_cli };
	const size_t gws[1] = { round_mul_up(nels, lws[0]) };

	cl_uint arg_index = 0;
	err = clSetKernelArg(init_k, arg_index, sizeof(cl_mem), &vec1);
	ocl_check(err, "init_k set kernel arg %u", arg_index);
	++arg_index;
	err = clSetKernelArg(init_k, arg_index, sizeof(nels), &nels);
	ocl_check(err, "init_k set kernel arg %u", arg_index);

	err = clEnqueueNDRangeKernel(que, init_k, 1, NULL, gws, lws, 0, NULL, &init_evt);

	ocl_check(err, "enqueue kernel init_k");
	return init_evt;
}

cl_event sum2(cl_command_queue que, cl_kernel sum_k,
	cl_mem out, cl_mem in, cl_int npairs, cl_int lws_cli, cl_event dep_evt)
{
	cl_int err;
	cl_event sum_evt;

	const size_t lws[1] = { lws_cli };
	const size_t gws[1] = { round_mul_up(npairs, lws[0]) };

	cl_uint arg_index = 0;
	err = clSetKernelArg(sum_k, arg_index, sizeof(cl_mem), &out);
	ocl_check(err, "sum_k set kernel arg %u", arg_index);
	++arg_index;
	err = clSetKernelArg(sum_k, arg_index, sizeof(cl_mem), &in);
	ocl_check(err, "sum_k set kernel arg %u", arg_index);
	++arg_index;
	err = clSetKernelArg(sum_k, arg_index, sizeof(cl_int), &npairs);
	ocl_check(err, "sum_k set kernel arg %u", arg_index);

	err = clEnqueueNDRangeKernel(que, sum_k, 1, NULL, gws, lws, 1, &dep_evt, &sum_evt);

	ocl_check(err, "enqueue kernel sum_k");
	return sum_evt;
}

void verify(const cl_int result, int nels) {
	int expected = nels*(nels+1)/2;
	if (expected != result) {
		fprintf(stderr, "%d != %d\n", expected, result);
		exit(7);
	}
}


int main(int argc, char *argv[])
{
	if (argc != 3) {
		fprintf(stderr, "%s nels vec lws\n", argv[0]);
		exit(1);
	}
	int nels = atoi(argv[1]);
	if (nels < 1) {
		fprintf(stderr, "nels deve essere almeno 1\n");
		exit(2);
	}
    int vec = atoi(argv[2]); // vettorizzazione decisa dall'utente
	if (vec & (vec - 1)) {
		fprintf(stderr, "vec deve essere una potenza di 2\n");
		exit(3);
	}
	if (vec < 2 || vec > 16) {
		fprintf(stderr, "vec deve essere compreso tra 2 e 16\n");
		exit(4);
	}
	int lws = atoi(argv[2]);
	if (lws < 1) {
		fprintf(stderr, "lws deve essere almeno 1\n");
		exit(5);
	}


	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("reduce.ocl", ctx, d);

	const size_t memsize = sizeof(cl_int)*nels;

	cl_int err;
	cl_mem d_vec1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, memsize, NULL, &err);
	ocl_check(err, "clCreateBuffer d_vec1 fallito");
	cl_mem d_vec2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, memsize/2, NULL, &err);
	ocl_check(err, "clCreateBuffer d_vec2 fallito");

	cl_kernel init_k = clCreateKernel(prog, "init_k", &err);
	ocl_check(err, "clCreateKernel init_k fallito");
	char sum_vk_name[9] = {0};
	snprintf(sum_vk_name, 8, "sum%d_k", vec);
	cl_kernel sum2_k = clCreateKernel(prog, sum_vk_name, &err);
	ocl_check(err, "clCreateKernel %s fallito", sum_vk_name);

	cl_event init_evt = init(que, init_k, d_vec1, nels, lws);

	int nvecs = 1;
	int steps = 1;
	for (; nvecs < nels ; ++steps, nvecs*=vec) {
	}
	if (nvecs > nels) {
		fprintf(stderr, "nels %d non è potenza di vec %d\n", nels, vec);
		exit(5);
	}
	cl_event *sum2_evt = calloc(steps, sizeof(cl_event));
	sum2_evt[0] = init_evt;

	cl_mem d_in = d_vec1;
	cl_mem d_out = d_vec2;
	nvecs = nels/vec;
    // Riduzione a coppie
	for (int i = 1 ; nvecs > 0 ; ++i, nvecs /= vec) {
		sum2_evt[i] = sum2(que, sum2_k, d_out, d_in, nvecs, lws, sum2_evt[i-1]);
		cl_mem t = d_in;
		d_in = d_out;
		d_out = t;
	}

	cl_event read_evt;
	cl_int risultato;
	err = clEnqueueReadBuffer(que, d_in, CL_TRUE, 0, sizeof(cl_int), &risultato, steps, sum2_evt, &read_evt);
	ocl_check(err, "read buffer d_vec");

	verify(risultato, nels);

	clFinish(que);

    /* Benchmarking */
	cl_ulong init_ns = runtime_ns(init_evt);
	printf("init: %.4gms %.4gGB/s %.4gGE/s\n", init_ns*1.0e-6, memsize/(double)init_ns, nels/(double)init_ns);

	cl_ulong reduction_ns = total_runtime_ns(sum2_evt[1], sum2_evt[steps-1]);
	printf("reduction: %.4gms %.4gGE/s\n", reduction_ns*1.0e-6, nels/(double)reduction_ns);

	nvecs = nels/vec;
	for (int i = 1 ; nvecs > 0 ; ++i, nvecs /= vec) {
		cl_ulong step_ns = runtime_ns(sum2_evt[i]);
		printf("step #%d: %.4gms %.4gGB/s %.4gGE/s\n", i,
			step_ns*1.0e-6,
			nvecs*(vec+1)*sizeof(cl_int)/(double)step_ns,
			nvecs/(double)step_ns);
	}

	cl_ulong read_ns = runtime_ns(read_evt);
	printf("read: %.4gms %.4gGB/s\n", read_ns*1.0e-6, sizeof(cl_int)/(double)read_ns);

    /* Cleanup */
    clReleaseEvent(init_evt);
    clReleaseEvent(read_evt);
    for (int i = 1; i < steps; ++i) {
        clReleaseEvent(sum2_evt[i]);
    }
    free(sum2_evt);
    clReleaseKernel(init_k);
    clReleaseKernel(sum2_k);
    clReleaseProgram(prog);
    clReleaseMemObject(d_vec1);
    clReleaseMemObject(d_vec2);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);

	return 0;
}