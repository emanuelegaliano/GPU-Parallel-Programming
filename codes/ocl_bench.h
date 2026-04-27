/* ocl_bench.h — OpenCL benchmark utilities: time and bandwidth measurement.
 *
 * Self-contained header: only requires <CL/cl.h>.
 * Include it alongside (or instead of) ocl_boiler.h — order does not matter.
 *
 * Usage example:
 *
 *   cl_event evt;
 *   clEnqueueNDRangeKernel(..., &evt);
 *   clFinish(que);
 *
 *   // single kernel: 2 arrays of n floats read + 1 written
 *   size_t bytes = 3 * n * sizeof(cl_float);
 *   ocl_bench_t b = ocl_bench_event(evt, bytes);
 *   ocl_bench_print("vecsum", b);
 *
 *   // range across multiple events (e.g. H2D + kernel + D2H)
 *   ocl_bench_t total = ocl_bench_range(evt_h2d, evt_d2h, bytes);
 *   ocl_bench_print("total", total);
 */

#ifndef OCL_BENCH_H
#define OCL_BENCH_H

#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#  include <OpenCL/cl.h>
#else
#  include <CL/cl.h>
#endif

/* -------------------------------------------------------------------------
 * Internal helper — error check without depending on ocl_boiler.h
 * ------------------------------------------------------------------------- */

static inline void _bench_check(cl_int err, const char *msg)
{
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ocl_bench error in '%s': %d\n", msg, err);
        exit(1);
    }
}

/* -------------------------------------------------------------------------
 * Result struct
 * ------------------------------------------------------------------------- */

typedef struct {
    cl_ulong runtime_ns;    /* kernel/event wall-time in nanoseconds          */
    double   runtime_ms;    /* same in milliseconds (convenience)             */
    double   bandwidth_GBs; /* effective bandwidth: bytes / runtime_ns [GB/s] */
    size_t   bytes;         /* total bytes transferred (read + written)       */
} ocl_bench_t;

/* -------------------------------------------------------------------------
 * Constructors
 * ------------------------------------------------------------------------- */

/* Build a bench result from a single profiling event and the total number
 * of bytes moved (reads + writes) by that event/kernel. */
static inline ocl_bench_t ocl_bench_event(cl_event evt, size_t bytes)
{
    cl_ulong start = 0, end = 0;
    _bench_check(
        clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START,
                                sizeof(cl_ulong), &start, (size_t *)0),
        "get start");
    _bench_check(
        clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END,
                                sizeof(cl_ulong), &end, (size_t *)0),
        "get end");

    ocl_bench_t b;
    b.bytes         = bytes;
    b.runtime_ns    = end - start;
    b.runtime_ms    = (double)b.runtime_ns * 1.0e-6;
    /* bandwidth in GB/s: bytes/nanoseconds == GB/s  (1 GB = 1e9 B) */
    b.bandwidth_GBs = (b.runtime_ns > 0)
                      ? (double)bytes / (double)b.runtime_ns
                      : 0.0;
    return b;
}

/* Build a bench result spanning from the START of event `from` to the END
 * of event `to`.  Useful for timing sequences of enqueued operations. */
static inline ocl_bench_t ocl_bench_range(cl_event from, cl_event to,
                                           size_t bytes)
{
    cl_ulong start = 0, end = 0;
    _bench_check(
        clGetEventProfilingInfo(from, CL_PROFILING_COMMAND_START,
                                sizeof(cl_ulong), &start, (size_t *)0),
        "range: get start");
    _bench_check(
        clGetEventProfilingInfo(to, CL_PROFILING_COMMAND_END,
                                sizeof(cl_ulong), &end, (size_t *)0),
        "range: get end");

    ocl_bench_t b;
    b.bytes         = bytes;
    b.runtime_ns    = end - start;
    b.runtime_ms    = (double)b.runtime_ns * 1.0e-6;
    b.bandwidth_GBs = (b.runtime_ns > 0)
                      ? (double)bytes / (double)b.runtime_ns
                      : 0.0;
    return b;
}

/* -------------------------------------------------------------------------
 * Printing
 * ------------------------------------------------------------------------- */

/* Print a one-line benchmark summary:
 *   <label>          <time_ms> ms  |  <bandwidth> GB/s  (<bytes> B)
 */
static inline void ocl_bench_print(const char *label, ocl_bench_t b)
{
    printf("%-20s  %10.4f ms  |  %8.3f GB/s  (%zu B)\n",
           label, b.runtime_ms, b.bandwidth_GBs, b.bytes);
}

/* Same as ocl_bench_print but writes to an arbitrary FILE*. */
static inline void ocl_bench_fprint(FILE *fp, const char *label, ocl_bench_t b)
{
    fprintf(fp, "%-20s  %10.4f ms  |  %8.3f GB/s  (%zu B)\n",
            label, b.runtime_ms, b.bandwidth_GBs, b.bytes);
}

/* -------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------- */

/* Bytes moved by a kernel that reads `nread` elements and writes `nwrite`
 * elements of `elem_size` bytes each. */
static inline size_t ocl_bench_bytes(size_t nread, size_t nwrite,
                                      size_t elem_size)
{
    return (nread + nwrite) * elem_size;
}

#endif /* OCL_BENCH_H */