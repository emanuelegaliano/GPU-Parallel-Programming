// Compilazione: gcc -o vecinit vecinit.c -lOpenCL

#include <stdio.h>
#include <stdlib.h>
#include "ocl_boiler.h"


cl_event init(cl_command_queue que, cl_kernel init_k, cl_mem vec, int nels)
{
	cl_int err;
	cl_event init_evt;

	const size_t gws[1] = { nels };

	cl_uint arg_index = 0;
	err = clSetKernelArg(init_k, arg_index, sizeof(cl_mem), &vec);
	ocl_check(err, "init_k set kernel arg %u", arg_index);

	/* Chiamata al kernel init_k */
	err = clEnqueueNDRangeKernel(
		que,      // coda dei comandi
		init_k,   // kernel
		1,        // dimensionalità griglia (nel nostro caso 1D)
		NULL,     // global work offset
		gws,      // global work size = quanti elementi totali lanciare
		NULL,     // local work size = come raggruppare gli elementi
		0, NULL,  // waiting list vuota (non si attende nulla)
		&init_evt // evento associato al lancio del kernel
	);

	ocl_check(err, "enqueue kernel init_k");
	return init_evt;
}

void verify(const int *vec, int nels) {
	for (int i = 0; i < nels; ++i) {
		int expected = i;
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
	if (argc != 2) {
		fprintf(stderr, "%s nels\n", argv[0]);
		exit(1);
	}
	int nels = atoi(argv[1]);
	if (nels < 1) {
		fprintf(stderr, "nels deve essere almeno 1\n");
		exit(2);
	}

	/* Selezione del device da utilizzare e creazione del contesto e della coda di comandi */
	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);

	/* Selezione e compilazione del codice device */
	cl_program prog = create_program("vecinit.ocl", ctx, d);

	const size_t memsize = sizeof(int)*nels;
	cl_int err;
	cl_mem d_vec = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, memsize, NULL, &err);
	ocl_check(err, "clCreateBuffer fallito");

	const char *kname = "init_k";
	cl_kernel init_k = clCreateKernel(prog, kname, &err);
	ocl_check(err, "clCreateKernel %s fallito", kname);

	cl_event init_evt = init(que, init_k, d_vec, nels);

	int *h_vec = malloc(sizeof(int)*nels);
	if (!h_vec) {
		fprintf(stderr, "allocazione fallita\n");
		exit(3);
	}

	cl_event wait_list[1] = { init_evt };
	cl_event read_evt;
	err = clEnqueueReadBuffer(
		que,          // coda su cui richiedere il comando
		d_vec,        // buffer da cui leggere i dati
		CL_TRUE,      // funzione bloccante (= sincrona)
		0, memsize,   // offset in byte e numero di byte da copiare
		h_vec,        // buffer host su cui copiare i dati
		1, wait_list, // numero di eventi nella waiting list, waiting list
		&read_evt     // evento generato da ReadBuffer
	);
	ocl_check(err, "read buffer d_vec");

	verify(h_vec, nels);

	/* Benchmarking - Calcolo tempo di esecuzione e banda passante */
	cl_ulong init_ns = runtime_ns(init_evt);
	cl_ulong read_ns = runtime_ns(read_evt);
	printf("init: %.4gms %.4gGB/s\n", init_ns*1.0e-6, memsize/(double)init_ns);
	printf("read: %.4gms %.4gGB/s\n", read_ns*1.0e-6, memsize/(double)read_ns);

	/* Rilascio eventi e risorse OpenCL */
    clReleaseEvent(init_evt);
    clReleaseEvent(read_evt);
    clReleaseKernel(init_k);
    clReleaseProgram(prog);
    clReleaseMemObject(d_vec);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);

    free(h_vec);

	return 0;
}
