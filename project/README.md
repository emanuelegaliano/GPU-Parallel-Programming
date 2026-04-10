# Monte Carlo GPU — Probabilità di Distruzione della Vegetazione da Colata Lavica

Simulazione Monte Carlo parallelizzata su GPU (OpenCL) per stimare la probabilità che una quantità critica di vegetazione di una data categoria venga distrutta da una colata lavica.

---

## Descrizione del problema

Dato un territorio rappresentato come una griglia $n \times n$, si dispone di:

- una **mappa di probabilità di invasione** $P[i][j] \in [0,1]$: stima la probabilità che la cella $(i,j)$ venga raggiunta dalla colata lavica
- una **mappa di habitat** $H[i][j] \in \{0,1\}$: indica la presenza ($1$) o assenza ($0$) di una specifica categoria di vegetazione nella cella $(i,j)$

Una mappa di habitat separata esiste per ciascuna categoria di vegetazione.

L'obiettivo è calcolare, per ogni categoria, la probabilità che la frazione di celle invase superi una **soglia critica** $\theta \in [0,1]$.

---

## Parametri

| Simbolo | Tipo | Descrizione |
|---|---|---|
| $P[i][j]$ | `float[n][n]` | Probabilità di invasione della cella $(i,j)$ |
| $H[i][j]$ | `int[n][n]` | Mappa di habitat binaria per una categoria ($0$/$1$) |
| $n$ | `int` | Dimensione della griglia |
| $\theta$ | `float` | Soglia critica di distruzione per la categoria |
| $R$ | `int` | Numero di run Monte Carlo (ordine: centinaia di migliaia) |

---

## Algoritmo

Il pipeline completo è una sequenza di due stadi **Map/Reduce**, entrambi eseguiti su GPU.

```
Kernel 1 — Preprocessing:
  MAP:    P[i][j], H[i][j]  →  P'[i][j] = P[i][j] · H[i][j]     (n² work-item)
  REDUCE: P'[i][j]          →  p_vec[k]  (stream compaction)      (n² → N_c)

Kernel 2 — Monte Carlo:
  MAP:    p_vec[k], x~U(0,1) →  invaded[r][k] = (x ≤ p_vec[k])  (R·N_c work-item)
  REDUCE: invaded[r][k]      →  over_threshold[r]                 (R·N_c → R)
  REDUCE: over_threshold[r]  →  prob                              (R → 1)
```

### Kernel 1 — Preprocessing (GPU)

**MAP**: ogni work-item calcola $P'[i][j] = P[i][j] \cdot H[i][j]$ su un NDRange di $n^2$ work-item.

**REDUCE** (stream compaction): le celle non nulle vengono compattate in un vettore denso `p_vec` di lunghezza $N_c$, dove $N_c$ è il numero di celle appartenenti alla categoria.

```
P (n×n)  ×  H (n×n)  →  P' (n×n)  →  p_vec (N_c)

Esempio (n=3):
P  = [[0.2, 0.5, 0.1],    H  = [[1, 0, 1],     P' = [[0.2, 0.0, 0.1],
      [0.8, 0.3, 0.6],          [0, 1, 0],             [0.0, 0.3, 0.0],
      [0.4, 0.7, 0.9]]          [1, 0, 0]]             [0.4, 0.0, 0.0]]

p_vec = [0.2, 0.1, 0.3, 0.4]   (N_c = 4)
```

### Kernel 2 — Monte Carlo (GPU)

Si lancia su un NDRange 2D di dimensioni $R \times N_c$.

Ogni work-item $(r, k)$ è responsabile di:
- run $r$ → `get_global_id(0)`
- cella $k$ → `get_global_id(1)`

**MAP** (invasione):

$$
\text{invaded}[r][k] = \begin{cases} 1 & \text{se } x \leq \text{p\_vec}[k],\ x \sim U(0,1) \\ 0 & \text{altrimenti} \end{cases}
$$

**REDUCE** (conteggio per run):

$$
\text{invaded\_count}[r] = \sum_{k=0}^{N_c - 1} \text{invaded}[r][k]
$$

$$
\text{over\_threshold}[r] = \begin{cases} 1 & \text{se } \dfrac{\text{invaded\_count}[r]}{N_c} > \theta \\ 0 & \text{altrimenti} \end{cases}
$$

**REDUCE** (probabilità finale):

$$
\text{prob} = \frac{1}{R} \sum_{r=0}^{R-1} \text{over\_threshold}[r]
$$

### Visualizzazione del NDRange (Kernel 2)

```
              k=0      k=1      k=2    ...   k=N_c-1
           p=0.2    p=0.1    p=0.3          p=0.4
           ───────  ───────  ───────        ───────
r=0    →   WI(0,0)  WI(0,1)  WI(0,2)  ...  WI(0,Nc-1)  →  reduce  →  0 o 1
r=1    →   WI(1,0)  WI(1,1)  WI(1,2)  ...  WI(1,Nc-1)  →  reduce  →  0 o 1
...
r=R-1  →   WI(R,0)  WI(R,1)  WI(R,2)  ...  WI(R,Nc-1)  →  reduce  →  0 o 1
                                                                          ↓
                                                                    riduzione finale
                                                                    prob = count / R
```

---

## Struttura della memoria

| Buffer | Dove vive | Dimensione | Note |
|---|---|---|---|
| `P`, `H` | device, global | $n^2 \cdot$ `sizeof(float/int)` | input, trasferiti da host una volta sola |
| `P'` | device, global | $n^2 \cdot$ `sizeof(float)` | output Kernel 1 MAP, input stream compaction |
| `p_vec` | device, constant/global | $N_c \cdot$ `sizeof(float)` | output Kernel 1; letto da tutti i work-item del Kernel 2 |
| `invaded` | device, local | $N_c \cdot$ `sizeof(int)$ per work-group` | non materializzato globalmente |
| `over_threshold` | device, global | $R \cdot$ `sizeof(int)` | risultato parziale per run |
| `count` | host | `sizeof(int)` | risultato finale, trasferito da device |

---

## Analisi dei rischi

### Algoritmica

| Rischio | Impatto | Mitigazione |
|---|---|---|
| Qualità del RNG su GPU | Alto | Usare un RNG solido e riproducibile |
| Convergenza statistica con $R$ piccolo | Alto | Validare la stabilità di `prob` al crescere di $R$ |
| Bias per $N_c = 0$ (habitat assente) | Medio | Gestire il caso degenere prima del lancio del kernel |
| Indipendenza spaziale delle celle assunta | Medio | Il modello non cattura la correlazione spaziale della colata; documentare il limite |

### Implementativa (OpenCL)

| Rischio | Impatto | Mitigazione |
|---|---|---|
| NDRange $R \times N_c$ troppo grande per il device | Alto | Suddividere $R$ in batch se necessario |
| Occupancy bassa per $N_c$ piccolo | Medio | Regolare la dimensione del work-group dinamicamente |
| Divergenza dei work-item nel confronto soglia | Basso | Usare predication invece di branch espliciti |
| Stato RNG non riproducibile tra run diversi | Basso | Seed deterministico funzione di $(r, k)$ |

### Di dominio

| Rischio | Impatto | Mitigazione |
|---|---|---|
| Soglia critica $\theta$ non calibrata | Alto | Va definita con esperti di ecologia/vulcanologia |
| Mappa $P$ statica (snapshot temporale) | Medio | Non considera evoluzione dinamica della colata |

---

## Dipendenze

- `ocl_boiler.h` (utility wrapper OpenCL)

---

## Utilizzo

```bash
# TODO: aggiornare dopo implementazione
./monte_carlo <mappa_P> <mappa_H> <soglia> <n_run>
```
