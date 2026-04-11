kernel void init_k(global int * restrict vec1, global int * restrict vec2, int nels)
{
	int i = get_global_id(0);
	if (i >= nels) return;
	vec1[i] = i;
	vec2[i] = nels - i;
}

// indici nell'array:    0 1 2 3 4 5 6 7 ...
// indici del work-item: 0 1 2 3 4 5 6 7 ...
kernel void sum_k(global int * restrict out,
	global int const * restrict vec1,
	global int const * restrict vec2,
	int nels)
{
	int i = get_global_id(0);
	if (i >= nels) return;
	out[i] = vec1[i] + vec2[i];
}

// indici nell'array:    0 1 2 3 4 5 6 7 ...
// indici del work-item: 0 0 1 1 2 2 3 3 ...
kernel void sum_s2_k(global int * restrict out,
	global int const * restrict vec1,
	global int const * restrict vec2,
	int nels)
{
	int wi = get_global_id(0);
	int gi0 = 2*wi;
	int gi1 = 2*wi + 1;

	if (gi0 < nels) out[gi0] = vec1[gi0] + vec2[gi0];
	if (gi1 < nels) out[gi1] = vec1[gi1] + vec2[gi1];
}

kernel void sum_s4_k(global int * restrict out,
	global int const * restrict vec1,
	global int const * restrict vec2,
	int nels)
{
	int wi = get_global_id(0);
	int gi0 = 4*wi;
	int gi1 = 4*wi + 1;
	int gi2 = 4*wi + 2;
	int gi3 = 4*wi + 3;

	if (gi0 < nels) out[gi0] = vec1[gi0] + vec2[gi0];
	if (gi1 < nels) out[gi1] = vec1[gi1] + vec2[gi1];
	if (gi2 < nels) out[gi2] = vec1[gi2] + vec2[gi2];
	if (gi3 < nels) out[gi3] = vec1[gi3] + vec2[gi3];
}


// indici nell'array:    0 1 2 3 4 5 6 7 ...
// indici del work-item: 0 1 2 3 0 1 2 3 ...
kernel void sum_s2s_k(global int * restrict out,
	global int const * restrict vec1,
	global int const * restrict vec2,
	int nels)
{
	int wi = get_global_id(0);
	int gi0 = wi;
	int gi1 = wi + get_global_size(0);

	if (gi0 < nels) out[gi0] = vec1[gi0] + vec2[gi0];
	if (gi1 < nels) out[gi1] = vec1[gi1] + vec2[gi1];
}

kernel void sum_s4s_k(global int * restrict out,
	global int const * restrict vec1,
	global int const * restrict vec2,
	int nels)
{
	const int gws = get_global_size(0);
	int wi = get_global_id(0);
	int gi0 = wi;
	int gi1 = wi +   gws;
	int gi2 = wi + 2*gws;
	int gi3 = wi + 3*gws;
#if 0
	int gi1 = gi0 + gws;
	int gi2 = gi1 + gws;
	int gi3 = gi2 + gws;
#endif

	if (gi0 < nels) out[gi0] = vec1[gi0] + vec2[gi0];
	if (gi1 < nels) out[gi1] = vec1[gi1] + vec2[gi1];
	if (gi2 < nels) out[gi2] = vec1[gi2] + vec2[gi2];
	if (gi3 < nels) out[gi3] = vec1[gi3] + vec2[gi3];
}

kernel void sum_s4sx_k(global int * restrict out,
	global int const * restrict vec1,
	global int const * restrict vec2,
	int nels)
{
	const int gws = get_global_size(0);
	const int wi = get_global_id(0);
	const int gi0 = wi;
	const int gi1 = wi +   gws;
	const int gi2 = wi + 2*gws;
	const int gi3 = wi + 3*gws;
#if 0
	int gi1 = gi0 + gws;
	int gi2 = gi1 + gws;
	int gi3 = gi2 + gws;
#endif
	const int v10 = gi0 < nels ? vec1[gi0] : 0;
	const int v20 = gi0 < nels ? vec2[gi0] : 0;
	const int v11 = gi1 < nels ? vec1[gi1] : 0;
	const int v21 = gi1 < nels ? vec2[gi1] : 0;
	const int v12 = gi2 < nels ? vec1[gi2] : 0;
	const int v22 = gi2 < nels ? vec2[gi2] : 0;
	const int v13 = gi3 < nels ? vec1[gi3] : 0;
	const int v23 = gi3 < nels ? vec2[gi3] : 0;

	if (gi0 < nels) out[gi0] = v10 + v20;
	if (gi1 < nels) out[gi1] = v11 + v21;
	if (gi2 < nels) out[gi2] = v12 + v22;
	if (gi3 < nels) out[gi3] = v13 + v23;
}

// indici nell'array:    0 1 2 3 4 5 6 7 ...
// indici del work-item: 0 0 1 1 2 2 3 3 ...
kernel void sum_v2_k(global int2 * restrict out,
	global int2 const * restrict vec1,
	global int2 const * restrict vec2,
	int pairs)
{
	int wi = get_global_id(0);
	if (wi >= pairs) return;
	out[wi] = vec1[wi] + vec2[wi];
}

kernel void sum_v4_k(global int4 * restrict out,
	global int4 const * restrict vec1,
	global int4 const * restrict vec2,
	int quarts)
{
	int wi = get_global_id(0);
	if (wi >= quarts) return;
	out[wi] = vec1[wi] + vec2[wi];
}

kernel void sum_v8_k(global int8 * restrict out,
	global int8 const * restrict vec1,
	global int8 const * restrict vec2,
	int octets)
{
	int wi = get_global_id(0);
	if (wi >= octets) return;
	out[wi] = vec1[wi] + vec2[wi];
}

kernel void sum_v16_k(global int16 * restrict out,
	global int16 const * restrict vec1,
	global int16 const * restrict vec2,
	int nhex) // nels/16
{
	int wi = get_global_id(0);
	if (wi >= nhex) return;
	out[wi] = vec1[wi] + vec2[wi];
}


// indici nell'array:    0 1 2 3 4 5 6 7 ...
// indici del work-item: 0 1 2 3 0 1 2 3 ...
kernel void sum_sliding_k(global int * restrict out,
	global int const * restrict vec1,
	global int const * restrict vec2,
	int nels)
{
	int wi = get_global_id(0);
	while (wi < nels) {
		out[wi] = vec1[wi] + vec2[wi];
		wi += get_global_size(0);
	}
}
