kernel void init_k(global int * restrict vec, int nels)
{
	int i = get_global_id(0);
	if (i >= nels) return;
	vec[i] = i;
}

kernel void smooth_k(global int * restrict out, global const int * restrict in, int nels)
{
	int i = get_global_id(0);
	if (i >= nels) return;

    int acc = in[i]; // accumulatore -> valore che c'è sicuramente
    int div = 1; // coefficiente per cui dividere -> c'è sicuramente 1 valore

    // se c'è un elemento precedente (non sono il 1° elemento)
    if (i > 0){
        acc += in[i-1];
        ++div;
    }
    // se c'è un elemento successivo
    if (i < nels - 1){
        acc += in[i+1];
        ++div;
    }

    out[i] = acc/div;
}

// Supponiamo che il numero di elementi sia almeno 4
kernel void smooth_v4_k(global int4 * restrict out, global const int4 * restrict in, int nquarts)
{
	int i = get_global_id(0);
	if (i >= nquarts) return;

    const int4 val = in[i]; // .x .y .z .w
    // acc = valore + valori precedenti (già presenti) + valori successivi (già presenti) 
    int4 acc = val + (int4)(0, val.xyz) + (int4)(val.yzw, 0);
    int4 div = (int4)(2,3,3,2);

    if(i > 0) {
        // elemento precedente della quartina corrente == quarto elemento della quartina precedente
        // usando il puntatore ci assicuriamo che non venga letta l'intera quartina precedente ma sollo l'elemento che ci interessa
        acc.x += *((global const int*)(in + i) - 1); // equivale a in[i-1].w
        div.x += 1;
    }
    if(i < nquarts - 1) {
        acc.w += *((global const int*)(in + i + 1)); // equivale a in[i+1].x;
        div.w += 1;
    }

    out[i] = acc/div;
}


kernel void smooth_lmem_k(global int * restrict out, global const int * restrict in, int nels, local int * restrict cache) // supponiamo dimensione cache = lws+2 (definita dall'host)
{
    const int lws = get_local_size(0);
	int gi = get_global_id(0);
    int li = get_local_id(0);

    int val;
    int acc;
    int div = 1;

	if (gi < nels){
        int val = in[gi];
        // inizializzazione elemento centrale neòòa cache
        cache[li+1] = val;

        // il 1° work item del work group carica il proprio predecessore nel 1° elemento della cache
        // gi != 0 -> controllo che il predecessore esista (non è il primo work item del work group)
        if(li == 0 && gi != 0){
            cache[0] = in[gi-1];
        }
        // l'ultimo work item carica il proprio successore nell'ultimo elemento della cache
        // gi < nels - 1 -> controllo che il successore esista (non è l'ultimo work item del work group)
        if(li == lws-1 && gi < nels - 1){
            cache[li+2] = in[gi+1];
        }
        
        acc = val;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(gi >= nels) return;

    if (gi > 0){
        acc += cache[li];
        ++div;
    }
    if (gi < nels - 1){
        acc += cache[li+2];
        ++div;
    }

    out[gi] = acc/div;
}


// Ogni cache contiene lws elementi
kernel void smooth_v4_lmem_k(global int4 * restrict out, global const int4 * restrict in, int nquarts, local int * restrict cache_prec, local int * restrict cache_succ)
{
	int gi = get_global_id(0);
    int li = get_local_id(0);
    const int lws = get_local_size(0);

    // carichiamo in[gi] solo se c'è effettivamente il dato
    const int4 val = gi < nquarts ? in[gi] : (int4)(0);

    if(li + 1 < lws) cache_prec[li+1] = val.w;
    if(li > 0) cache_succ[li-1] = val.x;

    // primo work item -> recupera il suo elemento precedente dalla global memory
    if(gi > 0 && li == 0){
        cache_prec[0] = *((global const int*)(in + gi) - 1);
    }
    // ultimo elemento -> recupera il suo elemento successivo
    if(gi < nquarts - 1 && li == lws - 1){
        cache_succ[li] = *((global const int*)(in + gi + 1));
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    // valore + componenti precedenti + componenti successive
    int4 acc = val + (int4)(0, val.xyz) + (int4)(val.yzw, 0);
    int4 div = (int4)(2,3,3,2);

    if(gi > 0) {
        acc.x += cache_prec[li];
        div.x += 1;
    }

    if(gi < nquarts - 1) {
        acc.w += cache_succ[li];
        div.w += 1;
    }
    
    if(gi < nquarts) out[gi] = acc/div;
}