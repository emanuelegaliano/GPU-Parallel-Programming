#include <stdio.h>
#include <stdlib.h>
#include "ocl_boiler.h"

cl_event init(cl_command_queue que, cl_kernel init_k, cl_mem vec, int nels){
	cl_int err;
	cl_event init_evt;
	const size_t gws[1] = { nels }; 		
	cl_uint arg_index = 0;
	err = clSetKernelArg(init_k, arg_index, sizeof(cl_mem), &vec);
	ocl_check(err, "init_k set kernel arg");
	
	err = clEnqueueNDRangeKernel(que, init_k, 1, NULL, gws, NULL, 0, NULL, &init_evt);
	ocl_check(err, "enqueue kernel init_k");
	return init_evt;
}


void verify(const int *vec, int nels){
	for(int i = 0; i < nels; ++i){
		int expected = i;
		int computed = vec[i];
		if(expected != computed){
			fprintf(stderr, "%d != %d @ %d\n", expected, computed, i);
			exit(4);
		}
	}
}


int main(int argc, char **argv){
	if(argc != 2){
		fprintf(stderr, "%s nels\n", argv[0]);
		exit(1);
	}
	
	int nels = atoi(argv[1]);
	if(nels < 1){
		fprintf(stderr, "nels deve essere almeno 1\n");
		exit(2);
	}
	
	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	
	cl_program prog = create_program("vecinit.ocl", ctx, d);
	
	const size_t memsize = sizeof(int)*nels;
	
	cl_int err;
	/* Allocazione del buffer host da parte del device */
	cl_mem d_vec = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, memsize, NULL, &err);
	ocl_check(err, "clCreateBuffer fallito");
	
	const char *kname = "init_k";
	cl_kernel init_k = clCreateKernel(prog, kname, &err);
	ocl_check(err, "clCreateKernel %s fallito", kname);
	
	cl_event init_evt = init(que, init_k, d_vec, nels);
	cl_event wait_list[1] = { init_evt }; // waiting list
	cl_event map_evt;
	
	/* Mappatura e restituzione del puntatore al buffer host */
	cl_int *h_vec = clEnqueueMapBuffer(que,          // coda su cui richiedere il comando
									   d_vec,        // buffer da cui leggere i dati
									   CL_TRUE,      // chiamata bloccante
									   CL_MAP_READ,  // access flag (in questo caso, mappatura in sola lettura)
									   0,            // offset (in byte) all'interno del buffer device,
									   memsize,      // numero di byte da mappare
									   1, wait_list, // gestione waiting list
									   &map_evt,     // evento associato
									   &err          // error code associato
	);
	ocl_check(err, "clEnqueueMapBuffer fallito");
	
	verify(h_vec, nels);
	
	/* Unmapping */
	cl_event unmap_evt;
	err = clEnqueueUnmapMemObject(que,       // coda dei comandi
		       					  d_vec,	 // buffer device
								  h_vec,	 // buffer host
								  0, NULL,   // gestione waiting list
								  &unmap_evt // evento associato
	);
	ocl_check(err, "clEnqueueUnmapMemObject fallito");
	
	/* Sincronizzazione */
	clFinish(que);
	
	/* Benchmarking */
	cl_ulong init_ns = runtime_ns(init_evt);
	cl_ulong map_ns = runtime_ns(map_evt);
	cl_ulong unmap_ns = runtime_ns(unmap_evt);
	printf("init: %.4gms %.4gGB/s\n", init_ns*1.0e-6, memsize/(double)init_ns);
	printf("map: %.4gms %.4gGB/s\n", map_ns*1.0e-6, memsize/(double)map_ns);
	printf("unmap: %.4gms %.4gGB/s\n", unmap_ns*1.0e-6, memsize/(double)unmap_ns);

	/* Rilascio eventi e risorse OpenCL */
    clReleaseEvent(init_evt);
    clReleaseEvent(map_evt);
    clReleaseEvent(unmap_evt);
    clReleaseKernel(init_k);
    clReleaseProgram(prog);
    clReleaseMemObject(d_vec);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);
	
	return 0;
}
