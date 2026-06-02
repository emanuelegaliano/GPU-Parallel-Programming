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

cl_event scan(cl_command_queue que, cl_kernel scan_k,
	cl_mem out, cl_mem code, cl_mem in, cl_int npairs, int nwg, cl_int lws_cli, cl_event dep_evt)
{
	cl_int err;
	cl_event scan_evt;

	const size_t lws[1] = { lws_cli };
	const size_t gws[1] = { nwg*lws[0] };

	cl_uint arg_index = 0;
	err = clSetKernelArg(scan_k, arg_index, sizeof(cl_mem), &out);
	ocl_check(err, "scan_k set kernel arg %u", arg_index);
	++arg_index;
	err = clSetKernelArg(scan_k, arg_index, sizeof(cl_mem), &code);
	ocl_check(err, "scan_k set kernel arg %u", arg_index);
	++arg_index;
	err = clSetKernelArg(scan_k, arg_index, sizeof(cl_mem), &in);
	ocl_check(err, "scan_k set kernel arg %u", arg_index);
	++arg_index;
	err = clSetKernelArg(scan_k, arg_index, lws[0]*sizeof(cl_int), NULL);
	ocl_check(err, "scan_k set kernel arg %u", arg_index);
	++arg_index;
	err = clSetKernelArg(scan_k, arg_index, sizeof(cl_int), &npairs);
	ocl_check(err, "scan_k set kernel arg %u", arg_index);

	err = clEnqueueNDRangeKernel(que, scan_k, 1, NULL, gws, lws, 1, &dep_evt, &scan_evt);

	ocl_check(err, "enqueue kernel scan_k");

	return scan_evt;
}

cl_event correction(cl_command_queue que, cl_kernel correction_k,
	cl_mem out, cl_mem code, cl_int npairs, int nwg, cl_int lws_cli, cl_event dep_evt)
{
	cl_int err;
	cl_event correction_evt;

	const size_t lws[1] = { lws_cli };
	const size_t gws[1] = { nwg*lws[0] };

	cl_uint arg_index = 0;
	err = clSetKernelArg(correction_k, arg_index, sizeof(cl_mem), &out);
	ocl_check(err, "correction_k set kernel arg %u", arg_index);
	++arg_index;
	err = clSetKernelArg(correction_k, arg_index, sizeof(cl_mem), &code);
	ocl_check(err, "correction_k set kernel arg %u", arg_index);
	++arg_index;
	err = clSetKernelArg(correction_k, arg_index, sizeof(cl_int), &npairs);
	ocl_check(err, "correction_k set kernel arg %u", arg_index);

	err = clEnqueueNDRangeKernel(que, correction_k, 1, NULL, gws, lws, 1, &dep_evt, &correction_evt);

	ocl_check(err, "enqueue kernel correction_k");

	return correction_evt;
}

void verify(const cl_int *risultato, int nels) {
	int atteso = 0;
	for (int i = 0; i < nels; ++i) {
		atteso += i+1;
		if (atteso != risultato[i]) {
			fprintf(stderr, "%d != %d @ %d\n", atteso, risultato[i], i);
			exit(7);
		}
	}
}


int main(int argc, char *argv[])
{
	if (argc != 5) {
		fprintf(stderr, "%s nels vec lws nwg_cu\n", argv[0]);
		exit(1);
	}
	int nels = atoi(argv[1]);
	if (nels < 1) {
		fprintf(stderr, "nels deve essere almeno 1\n");
		exit(2);
	}
	int vec = atoi(argv[2]);
	if (vec != 4) {
		fprintf(stderr, "vec deve essere 4\n");
		exit(2);
	}
	int lws = atoi(argv[3]);
	if (lws < 1) {
		fprintf(stderr, "lws deve essere almeno 1\n");
		exit(2);
	}
	// Work groups per compute unit (se positivo) / Work groups totali (se negativo)
	int nwg_cu = atoi(argv[4]);
	if (nwg_cu == 0) {
		fprintf(stderr, "nwg_cu deve essere almeno 1 (negativo per numero di workgroup totali)\n");
		exit(2);
	}
	if (nels & (vec - 1)) {
		fprintf(stderr, "nels %d deve essere multiplo di nvec %d\n", nels, vec);
		exit(2);
	}

	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);

	// Compute units
	cl_int cu;
	ocl_check(clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL), "get max compute_units");

	// Numero di work groups totali
	int nwg = nwg_cu < 0 ? -nwg_cu : nwg_cu * cu;
	if (nwg > 1) {
		nwg = round_mul_up(nwg, vec);
	}
	printf("%d wg per CU, %d CU = %d work-groups\n", nwg_cu, cu, nwg);

	cl_program prog = create_program("scan.ocl", ctx, d);

	const size_t memsize = sizeof(cl_int)*nels;

	cl_int err;
	cl_mem d_original = clCreateBuffer(ctx, CL_MEM_READ_WRITE, memsize, NULL, &err);
	ocl_check(err, "clCreateBuffer d_original fallito");
	cl_mem d_scan_result = clCreateBuffer(ctx, CL_MEM_READ_WRITE, memsize, NULL, &err);
	ocl_check(err, "clCreateBuffer d_scan_result fallito");
	cl_mem d_code = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nwg*sizeof(cl_int), NULL, &err);
	ocl_check(err, "clCreateBuffer d_code fallito");

	cl_kernel init_k = clCreateKernel(prog, "init_k", &err);
	ocl_check(err, "clCreateKernel init_k fallito");

	char scan_vk_name[22] = {0};
	snprintf(scan_vk_name, 21, "scan_v%d_sliding_k", vec);
	char correction_vk_name[32] = {0};
	snprintf(correction_vk_name, 31, "scan_v%d_sliding_correction_k", vec);
	cl_kernel scan_k = clCreateKernel(prog, scan_vk_name, &err);
	ocl_check(err, "clCreateKernel %s fallito", scan_vk_name);
	cl_kernel correction_k = clCreateKernel(prog, correction_vk_name, &err);
	ocl_check(err, "clCreateKernel %s fallito", correction_vk_name);

	cl_event init_evt = init(que, init_k, d_original, nels, lws);

	// 3 step se il numero di wg è maggiore di 1 (scan + scan delle code + correzioni)
	// 1 step se si ha un unico wg (scan)
	const int steps = 1 + (nwg > 1 ? 3 : 1);
	cl_event scan_evt[4];
	scan_evt[0] = init_evt;

	scan_evt[1] = scan(que, scan_k, d_scan_result, d_code, d_original, nels/vec, nwg, lws, scan_evt[0]);
	if (nwg > 1) {
		scan_evt[2] = scan(que, scan_k, d_code, d_code, d_code, nwg/vec, 1, lws, scan_evt[1]);
		scan_evt[3] = correction(que, correction_k, d_scan_result, d_code, nels/vec, nwg, lws, scan_evt[2]);
	}

	cl_event read_evt;
	cl_int *risultato = clEnqueueMapBuffer(que, d_scan_result, CL_TRUE, CL_MAP_READ, 0, memsize, steps, scan_evt, &read_evt,
		&err);
	ocl_check(err, "Enqueue map buffer");

	verify(risultato, nels);

	clFinish(que);

	cl_ulong init_ns = runtime_ns(init_evt);
	printf("init: %.4gms %.4gGB/s %.4gGE/s\n", init_ns*1.0e-6,
		memsize/(double)init_ns,
		nels/(double)init_ns);

	cl_ulong scan_ns = total_runtime_ns(scan_evt[1], scan_evt[steps-1]);
	printf("scan: %.4gms %.4gGE/s\n", scan_ns*1.0e-6, nels/(double)scan_ns);

	cl_ulong step_ns = runtime_ns(scan_evt[1]);
	printf("step #%d: %.4gms %.4gGB/s %.4gGE/s\n", 1,
		step_ns*1.0e-6,
		(2*memsize + (nwg > 1 ? nwg : 0)*sizeof(cl_int))/(double)step_ns,
		nels/(double)step_ns);
	if (nwg > 1) {
		cl_ulong code_ns = runtime_ns(scan_evt[2]);
		printf("step #%d: %.4gms %.4gGB/s %.4gGE/s\n", 2,
			code_ns*1.0e-6,
			(2*nwg)*sizeof(cl_int)/(double)code_ns,
			nwg/(double)code_ns);
		cl_ulong correzione_ns = runtime_ns(scan_evt[3]);
		printf("step #%d: %.4gms %.4gGB/s %.4gGE/s\n", 3,
			correzione_ns*1.0e-6,
			2*memsize/(double)correzione_ns,
			nels/(double)correzione_ns);
	}

	cl_ulong read_ns = runtime_ns(read_evt);
	printf("read: %.4gms %.4gGB/s\n", read_ns*1.0e-6, memsize/(double)read_ns);

	return 0;
}