// kernel per preparare i dati
kernel void init_k(global int * restrict vec1, global int * restrict vec2, int nels)
{
	int i = get_global_id(0);
	if(i >= nels) return;
	vec1[i] = i;
	vec2[i] = nels - i;
}

// kernel per sommare i vettori
kernel void sum_k(global int * restrict out, global int const * restrict vec1, global int const * restrict vec2, int nels)
{
	int i = get_global_id(0);
	if(i >= nels) return;
	out[i] = vec1[i] + vec2[i];
}