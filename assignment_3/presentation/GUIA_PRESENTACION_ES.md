# Guía Maestra de Presentación Oral (3 a 5 minutos)
## Práctica 3: Paralelización en Memoria Compartida y Vectorización SIMD
**Asignatura:** Big Data (GCID) · ULPGC · Curso 2026–2027  
**Autor:** Pablo Martínez Suárez  
**Documento de referencia:** `Assignment03_Martinez_Report.pdf`  
**Diapositivas:** `presentation/Assignment03_Presentation.pdf` y visor web interactivo en `presentation/presentation.html`

---

## 1. Estrategia y Control del Tiempo (Time Budget)

La presentación está milimétricamente diseñada para durar **entre 3:20 y 3:45 minutos**, dejando más de un minuto de margen para el turno de defensa y preguntas sin exceder el límite de **5 minutos**.

| Diapositiva | Tema Central | Tiempo Objetivo | Minuto Acumulado |
| :--- | :--- | :---: | :---: |
| **Slide 1** | **Rigor Experimental y Microarquitectura** (Apple M2 4P+4E, SplitMix64, CLOCK_MONOTONIC, ausencia de data races) | 45 s | 0:00 – 0:45 |
| **Slide 2** | **Escalado Fuerte y Tres Regímenes** (Overhead en $p=1$, Ley de Amdahl, saturación y sobreasignación en $p=16$) | 60 s | 0:45 – 1:45 |
| **Slide 3** | **Heterogeneidad y Planificación Dinámica** (Cuello de botella estático en Big.LITTLE vs ganancia del +11.7% con `dynamic`) | 65 s | 1:45 – 2:50 |
| **Slide 4** | **Vectorización NEON, Muro de Memoria y Conclusiones** (Instrucción `fmla.2d`, saturación de ancho de banda y puente a P4) | 55 s | 2:50 – 3:45 |
| **Defensa / Q&A** | Preguntas del tribunal / profesor | ~1 min | 3:45 – 5:00 |

> [!TIP]
> **Regla de oro de la defensa:** No leas las diapositivas. Las diapositivas presentan los gráficos y métricas formales en inglés; tú debes narrar la **física computacional** del experimento: por qué $p=1$ es más lento que el C secuencial puro, por qué los núcleos de eficiencia provocan esperas en la barrera estática, y cómo la memoria compartida impone un techo insalvable que exige computación distribuida.

---

## 2. Guión Detallado Diapositiva a Diapositiva

---

### Diapositiva 1: Experimental Rigor & Protocol
*(Duración sugerida: 40 - 45 segundos | Minuto 0:00 a 0:45)*

#### 🎯 Mensaje central
> "Evaluamos el paralelismo en memoria compartida de forma rigurosa y controlada sobre una arquitectura asimétrica real (Apple M2), garantizando reproducibilidad determinista y ausencia total de condiciones de carrera."

#### 🗣️ Guión sugerido (Qué decir en español de forma natural):
> *"Buenos días. En esta tercera práctica abordamos la paralelización en memoria compartida mediante OpenMP y la vectorización SIMD explícita aplicada al producto de matrices denso.*
>
> *Nuestra pregunta central es: ¿cómo afecta el número de hilos $p$ (desde 1 hasta 16) al tiempo de ejecución, al speedup y a la eficiencia paralela cuando ejecutamos sobre un procesador heterogéneo como el Apple M2, compuesto por 4 núcleos de alto rendimiento y 4 de eficiencia?*
>
> *Para que la comparativa sea estrictamente válida según el protocolo común, el método paralelo utiliza exactamente el mismo algoritmo con bucles reordenados $i\text{-}k\text{-}j$, con acceso contiguo en memoria, comparándose contra una línea base secuencial idéntica compilada sin el runtime de OpenMP. Cada hilo es dueño exclusivo de la fila $i$ de la matriz resultado, garantizando cero condiciones de carrera sin necesidad de cerrojos. Las entradas se generan mediante SplitMix64 determinista, el cronómetro monotonic aísla solo el cálculo aritmético y descartamos iteraciones de calentamiento reportando mediana e IQR de 5 repeticiones."*

#### 🔑 Datos técnicos que debes pronunciar con soltura:
- **Plataforma:** Apple M2 (ARMv8.5-A), 8 núcleos físicos (4 P-cores *Avalanche* a 3.50 GHz + 4 E-cores *Blizzard* a 2.42 GHz), 8 GB de memoria unificada LPDDR5 (100 GB/s).
- **Rango evaluado:** 7 dimensiones desde $n=64$ hasta $n=2048$, y 7 configuraciones de hilos: $p \in \{1, 2, 4, 6, 8, 12, 16\}$.
- **Verificación:** Error numérico verificado $< 10^{-12}$ frente a NumPy BLAS y casos analíticos exactos ($A \times 0 = 0$, $A \times I = A$).

---

### Diapositiva 2: Strong Scaling & Architectural Regimes
*(Duración sugerida: 55 - 60 segundos | Minuto 0:45 a 1:45)*

#### 🎯 Mensaje central
> "El escalado fuerte revela tres regímenes microarquitectónicos nítidos: la penalización del runtime en $p=1$, el escalado en los P-cores, y el freno por asimetría y sobreasignación."

#### 🗣️ Guión sugerido:
> *"Al analizar las curvas de escalado fuerte en la Diapositiva 2, observamos tres fenómenos fundamentales:*
>
> *Primero, el caso de un solo hilo ($p=1$): la implementación OpenMP sufre una penalización del 47% al 53% respecto al secuencial puro ($S(1) \approx 0.50$). Este resultado responde exactamente al requerimiento de la guía docente: cuantifica el coste de inicialización del pool de hilos, la gestión de barreras y la pérdida de ciertas optimizaciones del compilador en regiones paralelas.*
>
> *Segundo, entre $p=2$ y $p=4$, entramos en el Régimen I de núcleos de rendimiento puro: el throughput escala linealmente de 6.2 a 22.8 GFLOPS en $n=1024$.*
>
> *Tercero, en el Régimen II ($p=6$ a $8$), los 4 núcleos de eficiencia entran en juego. Al tener menor IPC y menor frecuencia (2.42 GHz frente a 3.5 GHz), la ganancia marginal disminuye. Finalmente, en el Régimen III con $p > 8$, entramos en sobreasignación (oversubscription): con 16 hilos en 8 contextos de hardware, el tiempo en $n=2048$ empeora de 0.65 a 0.80 segundos debido al coste de context switching involuntario del sistema operativo."*

#### 🔑 Datos clave a destacar:
- **Overhead $p=1$:** $0.47\times$ a $0.53\times$ respecto al C secuencial puro.
- **Pico estático:** 29.6 GFLOPS a $p=8$ ($n=1024$).
- **Regresión por sobreasignación:** Empeoramiento del 22% en tiempo al pasar de $p=8$ a $p=16$ en $n=2048$.

---

### Diapositiva 3: Scheduling Strategies & Heterogeneity
*(Duración sugerida: 60 - 65 segundos | Minuto 1:45 a 2:50)*

#### 🎯 Mensaje central
> "La planificación estática fracasa en procesadores asimétricos porque los núcleos rápidos quedan ociosos en la barrera; la planificación dinámica recupera un +11.7% de rendimiento."

#### 🗣️ Guión sugerido:
> *"La Diapositiva 3 expone el hallazgo arquitectónico más relevante de nuestro estudio: el impacto de la heterogeneidad de los núcleos.*
>
> *Bajo planificación estática (`schedule(static)`), OpenMP reparte $n/p$ filas idénticas a cada hilo. En un chip homogéneo esto es óptimo, pero en el Apple M2 es ineficiente: los 4 núcleos Avalanche terminan sus bloques en la mitad de tiempo y quedan completamente bloqueados en la barrera implícita esperando a que los núcleos Blizzard finalicen sus filas lentas.*
>
> *Para resolverlo, evaluamos planificación dinámica con tamaño de bloque (`schedule(dynamic, 8)`). En este esquema, los hilos solicitan bloques de 8 filas a una cola compartida según van quedando libres. Como resultado, los P-cores procesan cerca del 70% de la matriz, mientras que los E-cores contribuyen sin penalizar a los demás.*
>
> *La evidencia empírica es contundente: en $n=1024$ con 8 hilos, la planificación dinámica eleva el rendimiento de 29.6 a 33.1 GFLOPS, logrando un incremento inmediato del +11.7% y un speedup global de 2.51x frente al C secuencial puro."*

#### 🔑 Conceptos clave a defender:
- **Causa física:** Desincronización en la barrera de sincronización (`omp barrier`) por asimetría de frecuencia e IPC.
- **Métrica:** Salto de 29.63 GFLOPS a 33.09 GFLOPS en $p=8$ con chunk size de 8 filas.

---

### Diapositiva 4: Vectorisation, Memory Limits & Takeaways
*(Duración sugerida: 50 - 55 segundos | Minuto 2:50 a 3:45)*

#### 🎯 Mensaje central
> "La vectorización SIMD NEON opera a nivel de instrucción, pero ambas técnicas chocan contra el muro de ancho de banda de memoria compartida, motivando el salto a la computación distribuida."

#### 🗣️ Guión sugerido:
> *"Finalmente, en la Diapositiva 4 examinamos la extensión de vectorización SIMD y las limitaciones físicas del sistema.*
>
> *Implementamos la vectorización explícita utilizando registros NEON de 128 bits con la instrucción `fmla.2d`, que ejecuta dos operaciones fused multiply-accumulate en coma flotante de 64 bits por ciclo, complementada con un desenrollado de bucle 4x.*
>
> *Comprobamos que la vectorización acelera el procesamiento aritmético, pero a medida que el tamaño crece ($n=2048$, donde las tres matrices ocupan 96 MB y desbordan con creces los 16 MB de caché L2 compartida), los 8 núcleos compiten por el bus de memoria unificada de 100 GB/s. Este fenómeno es el Muro de Memoria (Memory Wall): las unidades SIMD quedan subalimentadas de datos, estabilizando la eficiencia paralela.*
>
> *Como conclusiones directas para la ingeniería de Big Data:*
> 1. *Medir siempre $p=1$ para conocer la sobrecarga del framework.*
> 2. *Usar planificación dinámica en procesadores asimétricos.*
> 3. *Nunca sobreasignar hilos más allá de los contextos físicos.*
> *Y como puente hacia la Práctica 4: el paralelismo en memoria compartida tiene un límite físico infranqueable en un solo nodo; para matrices masivas, la única solución es la partición distribuida mediante frameworks como MPI o Apache Spark. Muchas gracias."*

---

## 3. Banco de Preguntas Difíciles del Profesor (Q&A Defense)

### P1: ¿Por qué OpenMP con 1 hilo ($p=1$) es más lento que el programa secuencial compilado sin OpenMP?
> **Respuesta exacta:**  
> *"El ejecutable compilado con `-fopenmp` vincula la biblioteca `libomp`, que al iniciar el proceso reserva estructuras de control del pool de hilos, inicializa barreras internas y sustituye el bucle exterior por llamadas a funciones de despacho en el runtime. Además, en el código puramente secuencial el compilador Clang puede aplicar ciertas optimizaciones interprocedimentales de registros que se inhiben cuando el bucle está marcado como sección paralela compartida."*

### P2: ¿Por qué en $p=16$ el rendimiento empeora respecto a $p=8$ o $p=12$?
> **Respuesta exacta:**  
> *"El procesador Apple M2 dispone de 8 núcleos físicos sin Hyper-Threading (SMT desactivado), con 8 contextos de ejecución hardware reales. Asignar 16 hilos obliga al planificador del kernel Darwin a realizar cambios de contexto involuntarios (preemptive context switches), expulsando líneas de las cachés L1D (64 KB) y L2 (16 MB) y aumentando el tiempo de espera por sincronización en las barreras."*

### P3: ¿Cómo demostraste que la vectorización realmente ocurre y no es una asunción?
> **Respuesta exacta:**  
> *"Lo demostramos en dos niveles: primero, mediante diagnósticos de optimización del compilador con el flag `-Rpass=loop-vectorize`, que reporta la vectorización del bucle con factor 2; y segundo, inspeccionando directamente el código ensamblador generado con `clang -S`, donde comprobamos la emisión explícita de instrucciones vectoriales `fmla.2d v, v, v[0]`, frente al ensamblador no vectorizado que emite instrucciones escalares individuales `fmadd d, d, d, d`."*

### P4: ¿Por qué no se producen condiciones de carrera (race conditions) en la matriz resultado $C$?
> **Respuesta exacta:**  
> *"Porque la paralelización se aplica sobre el bucle más externo $i$. La posición en memoria de cualquier celda $C[i \cdot n + j]$ depende exclusivamente del índice de fila $i$. Como cada fila $i$ es asignada de forma disjunta a un único hilo, ningún hilo escribe jamás en una posición asignada a otro. La matriz $A$ se lee por filas privadas y la matriz $B$ se comparte de forma estrictamente de solo lectura (read-only), evitando data races y false sharing ya que cada fila de doubles ocupa como mínimo 512 bytes (múltiplo de las líneas de caché de 64 bytes)."*

### P5: ¿Qué diferencia hay entre Speedup frente a $p=1$ y Speedup frente a la línea base secuencial?
> **Respuesta exacta:**  
> *"Es una distinción metodológica crucial que exige el PDF del curso: si medimos $S(p) = T(1) / T(p)$, usamos la versión paralela con 1 hilo como base, lo que oculta la ineficiencia del framework y puede dar números artificialmente altos (hasta 5.3x en $p=8$). Si calculamos el speedup riguroso frente a la línea base secuencial $S(p) = T_{seq} / T(p)$, el speedup pico es de 2.51x, reflejando el beneficio real de aceleración respecto a no usar paralelismo."*
