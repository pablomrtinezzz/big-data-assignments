# Guía Maestra de Presentación (2 a 5 minutos)
## Práctica 1: Multiplicación Básica de Matrices en C, Java y Python
**Asignatura:** Big Data (GICD) · ULPGC · Curso 2026–2027  
**Autor:** Pablo Martínez Suárez  
**Documento de referencia:** `Assignment01_Martinez_Report.pdf`  
**Diapositivas:** `presentation/Assignment01_Presentation.pdf` y visor web interactivo en `presentation/presentation.html`

---

## 1. Estrategia y Control del Tiempo (Time Budget)

La presentación está diseñada para durar **entre 3:15 y 3:45 minutos**, dejando hasta 1:15 minutos de margen de seguridad para no superar nunca el límite de **5 minutos**.

| Diapositiva | Tema Central | Tiempo Objetivo | Minuto Acumulado |
| :--- | :--- | :---: | :---: |
| **Slide 1** | **Rigor Experimental y Protocolo** (Hardware, PRNG, Monotonic Clocks, DCE) | 45 s | 0:00 – 0:45 |
| **Slide 2** | **Escalado Temporal y la Barrera de los 60s** ($O(n^3)$, Factor 189x, Timeout en $n=1536$) | 60 s | 0:45 – 1:45 |
| **Slide 3** | **Análisis Arquitectónico y Throughput** (Python vs C2 JIT vs C `-O2` vs `-O0`) | 65 s | 1:45 – 2:50 |
| **Slide 4** | **Memoria, Muro de Caché L2 y Conclusiones** (306 MB vs 24 MB, Stride de $B$, Práctica 2) | 55 s | 2:50 – 3:45 |
| **Cierre / Q&A** | Preguntas del profesor / Turno de defensa | ~1 min | 3:45 – 5:00 |

> [!TIP]
> **Regla de oro de la presentación oral:** No leas las diapositivas. Las diapositivas ya contienen los datos y gráficos limpios en inglés; tú debes contar la **historia del experimento**, destacar las **anomalías interesantes** y demostrar que dominas la **causa física/arquitectónica** de cada número.

---

## 2. Guión Detallado Diapositiva a Diapositiva

---

### Diapositiva 1: Experimental Rigor & Protocol
*(Duración sugerida: 40 - 45 segundos | Minuto 0:00 a 0:45)*

#### 🎯 Mensaje central
> "No es una comparativa informal; es un experimento científico estricto bajo condiciones idénticas de algoritmo, precisión y hardware."

#### 🗣️ Guión sugerido (Qué decir en español de forma natural):
> *"Buenos días. Hoy presento la evaluación experimental del producto de matrices canónico comparando tres paradigmas de ejecución: compilación nativa en C, máquina virtual con compilación JIT en Java (HotSpot), e interpretación dinámica en Python.*
>
> *Nuestra hipótesis de partida es: ¿cómo impacta la arquitectura del runtime al tiempo de ejecución y al consumo de memoria cuando mantenemos estrictamente idénticos el algoritmo $i\text{-}j\text{-}k$, la precisión de 64 bits en coma flotante y los datos de entrada?*
>
> *Para que la comparativa sea rigurosa y reproducible, implementamos el generador determinista SplitMix64 con semilla 42 en los tres lenguajes, garantizando matrices bit a bit equivalentes. El cronómetro aísla única y exclusivamente el bucle aritmético interno, ejecutamos 5 repeticiones reportando mediana e IQR, aplicamos rondas de calentamiento previas para estabilizar el compilador C2 de Java y calculamos un checksum final como defensa contra la eliminación de código muerto."*

#### 🔑 Datos técnicos que debes pronunciar con seguridad:
- **Plataforma:** Apple M2 (ARMv8.5-A), 8 núcleos (4 rendimiento a 3.5 GHz + 4 eficiencia), 8 GB memoria unificada LPDDR5, macOS arm64.
- **Rango de tamaños:** 10 dimensiones discretas desde $n = 32$ hasta $n = 2048$.
- **Límite operativo:** Timeout estricto de 60.0 segundos por repetición.
- **Verificación:** Error numérico relativo verificado $< 10^{-12}$ contra NumPy BLAS.

---

### Diapositiva 2: Execution Time Divergence & Feasibility Wall
*(Duración sugerida: 55 - 60 segundos | Minuto 0:45 a 1:45)*

#### 🎯 Mensaje central
> "La complejidad matemática $O(n^3)$ se cumple en todos los lenguajes, pero la constante de Python es 189 veces peor, provocando que choque contra el muro del timeout a partir de $n=1536$."

#### 🗣️ Guión sugerido:
> *"Pasando a los resultados temporales, en la gráfica de la izquierda vemos la comparación directa. En escala logarítmica (derecha), todas las curvas son paralelas a la asíntota cúbica. Al realizar la regresión lineal, encontramos exponentes alfa entre 3.10 y 3.38, lo que valida empíricamente que todos ejecutan la misma carga de trabajo $O(n^3)$.*
>
> *Sin embargo, el factor constante diverge dramáticamente: la constante $c$ de Python es **189 veces superior** a la de C optimizado.*
>
> *Si nos fijamos en una dimensión representativa como $n = 1024$: C optimizado tarda apenas **1.21 segundos**, Java tarda **1.76 segundos**, C sin optimizar sube a **3.80 segundos**, y Python se dispara hasta **41.15 segundos** (34 veces más lento que C).*
>
> *Y aquí alcanzamos un hallazgo clave: la **barrera de los 60 segundos**. Al evaluar $n = 1536$, la ley cúbica predecía 139 segundos para Python. Nuestro protocolo de seguridad abortó el proceso por timeout. Por tanto, el límite operativo viable para Python puro en este hardware es estrictamente $n = 1024$, mientras que C y Java escalan con soltura hasta $n = 2048$."*

#### 🔑 Datos técnicos que debes pronunciar:
- **Tiempos en $n=1024$:** C `-O2` = **1.21s** | Java = **1.76s** | C `-O0` = **3.80s** | Python = **41.15s**.
- **Factor de sobrecoste basal ($c_{Python} / c_{C-O2}$):** **189x**.
- **Timeout en $n=1536$:** Python abortado ($>60$s, proyectado en ~139s). $n_{\max}$ viable = 1024.
- **Exponente de C `-O2`:** $\alpha = 3.38$ ($R^2 = 0.995$), debido a penalizaciones de latencia de memoria en tamaños grandes.

---

### Diapositiva 3: Throughput Analysis & The Power of JIT Compilation
*(Duración sugerida: 60 - 65 segundos | Minuto 1:45 a 2:50)*

#### 🎯 Mensaje central
> "El rendimiento no depende solo del lenguaje: en Python el boxing dinámico destruye la caché, mientras que el compilador JIT de Java alcanza paridad con C gracias a vectorización NEON y desenrollado de bucles."

#### 🗣️ Guión sugerido:
> *"¿Por qué vemos esta divergencia tan extrema en throughput? Analicemos qué ocurre a nivel de arquitectura en el gráfico superior.*
>
> *Python se queda estancado en un techo de apenas **0.06 GFLOPS**, lo que representa menos del 1.5% de la capacidad de C. La razón es doble: primero, la sobrecarga del bucle de interpretación en `ceval.c`, donde cada operación aritmética requiere decodificar opcodes dinámicamente; y segundo, el **boxing de objetos**. En Python, cada número real no es un valor crudo en un registro, sino un `PyFloatObject` de 24 bytes en el heap. Para cada elemento $A[i][k]$ o $B[k][j]$, la CPU sufre un doble salto de punteros, destruyendo la predicción de saltos y la tubería de ejecución.*
>
> *En el otro extremo, el resultado más sorprendente del estudio es **Java HotSpot C2**. Tras las iteraciones de calentamiento, Java no solo compite de tú a tú con C `-O2`, sino que en $n = 2048$ llega a superarlo, completando la multiplicación en **19.1 segundos** frente a los **29.0 segundos** de C. El desensamblado del bytecode confirma que el compilador C2 desenrolló el bucle e introdujo instrucciones vectoriales NEON FMA (Fused Multiply-Accumulate).*
>
> *Por último, la comparativa entre C `-O2` y `-O0` revela un factor de aceleración del compilador de **3.15x**. En `-O0`, la variable acumuladora se escribe en la pila de memoria en cada iteración del bucle interno. Pero fíjense en este detalle crucial: incluso C sin optimizar es **10.8 veces más rápido que Python**, demostrando que la ejecución nativa sin boxing ofrece una ventaja estructural insalvable para un intérprete."*

#### 🔑 Datos técnicos que debes pronunciar:
- **Throughput:** C alcanza picos de 5.0 GFLOPS en tamaños pequeños y se estabiliza en ~1.8 GFLOPS. Python no supera nunca **0.06 GFLOPS**.
- **Java vs C en $n=2048$:** Java (19.09s) supera a C `-O2` (29.00s) gracias al runtime C2 y optimizaciones NEON.
- **Efecto calentamiento (Warm-up):** La primera ejecución en frío de Java tardó 3.5 veces más por la sobrecarga del tier 0 y profiling C1.
- **C `-O0` vs `-O2`:** 3.15x de diferencia por volcado continuo del acumulador (`sum`) a stack RAM.

---

### Diapositiva 4: Memory Overhead & Practical Takeaways
*(Duración sugerida: 50 - 55 segundos | Minuto 2:50 a 3:45)*

#### 🎯 Mensaje central
> "El diseño en memoria define tanto el consumo físico (Python infla 13 veces el payload) como los cuellos de botella de hardware (el salto por columnas desborda la caché L2)."

#### 🗣️ Guión sugerido:
> *"Para cerrar, analicemos la memoria y las implicaciones para la ingeniería de Big Data.*
>
> *Teóricamente, tres matrices de flotantes de doble precisión ocupan $24 n^2$ bytes. En $n = 1024$, esto representa exactamente **24.0 MB**. Como vemos en la gráfica, C y Java se ajustan casi a la perfección al payload teórico (**25.3 MB** de RSS en C y **23.8 MB** de heap en Java).*
>
> *En cambio, en Python el consumo del proceso se dispara hasta **306.3 MB**, lo que supone una expansión de **12.8 veces**. Esto ocurre porque cada flotante encapsulado más su puntero en la lista requiere 32 bytes (4 veces el tamaño de un double primitivo), sumado a la fragmentación de memoria y a las arenas del asignador `pymalloc`.*
>
> *A esto se suma el **muro de la caché L2**: en el orden de bucles $i\text{-}j\text{-}k$, la matriz $B$ se recorre por columnas. En una disposición 'row-major', cada acceso salta $8n$ bytes. En $n = 2048$, la matriz $B$ sola ocupa 32 MB, desbordando por completo la caché L2 unificada de 16 MB del chip M2, degradando el rendimiento a velocidad de DRAM.*
>
> *Las conclusiones prácticas para Big Data son contundentes:*
> 1. *Nunca implementar bucles numéricos internos en Python puro; recurrir a librerías nativas con BLAS o runtimes con JIT.*
> 2. *Java HotSpot es perfectamente viable para computación intensiva si se controlan las estructuras de datos y el calentamiento.*
> 3. *La física del hardware manda: la complejidad $O(n^3)$ en papel no te protege de la jerarquía de cachés.*
> 4. *Y como puente hacia la Práctica 2: reordenaremos el bucle a $i\text{-}k\text{-}j$ y aplicaremos cache-tiling para romper definitivamente este muro de memoria. Muchas gracias."*

#### 🔑 Datos técnicos que debes pronunciar:
- **Payload teórico $n=1024$:** 24.0 MB ($3 \times 8$ MB).
- **Consumo real RSS:** C (25.3 MB) vs Python (**306.3 MB** $\rightarrow$ 12.8x de inflación).
- **Caché L2 del Apple M2:** 16 MB compartida. A $n=2048$, una sola matriz ocupa 32 MB $\rightarrow$ fallos de caché continuos por paso de columna.
- **Próximo paso:** Algoritmos conscientes de la caché (tiling) y reordenación $i\text{-}k\text{-}j$ en el Assignment 2.

---

## 3. Banco de Preguntas y Respuestas Típicas (Q&A de Defensa)

Si el profesor o los evaluadores te hacen preguntas al finalizar los 4 minutos, aquí tienes las respuestas directas y fundamentadas en los datos del reporte:

---

### Pregunta 1: *"¿Por qué en $n=2048$ Java tardó 19 segundos y C tardó 29 segundos si C es lenguaje nativo?"*
> **Tu respuesta:**  
> *"Es una excelente observación. Ambos códigos implementan el bucle naive $i\text{-}j\text{-}k$. Sin embargo, en Clang con `-O2`, el compilador estático no reordena los bucles ni vectoriza agresivamente el acceso por columnas de $B$ para no alterar la asociatividad en coma flotante.  
> En cambio, la máquina virtual OpenJDK 22 con el compilador HotSpot C2 realiza optimizaciones en tiempo de ejecución tras perfilar el código: aplica desenrollado de bucles (loop unrolling) y emite instrucciones vectoriales NEON fused multiply-accumulate (FMA) para registros de 128 bits en ARM64. Además, en Java usamos un array de arrays (`double[][]`), lo que fragmenta ligeramente las filas en el heap y paradójicamente alteró la distribución de líneas de caché frente al array 1D continuo de C cuando la matriz superó los 16 MB de la caché L2."*

---

### Pregunta 2: *"¿Por qué Python escala con exponente 3.10 igual que los demás pero tarda 34 veces más?"*
> **Tu respuesta:**  
> *"Porque la complejidad asintótica $O(n^3)$ solo describe el crecimiento relativo con respecto a $n$, no el factor constante $c$.  
> En Python, la constante $c$ es **189 veces mayor** ($1.91 \times 10^{-8}$ frente a $1.01 \times 10^{-10}$ en C). Por cada una de las $2n^3$ operaciones flotantes, el bucle de evaluación de CPython (`ceval.c`) tiene que despachar opcodes de bytecode, resolver tipos dinámicamente y seguir dos punteros en memoria para desempaquetar el valor double desde un objeto `PyFloatObject`. Hacen falta decenas de instrucciones máquina por cada FLOP en Python, mientras que C emite una única instrucción ensamblador."*

---

### Pregunta 3: *"¿Por qué hubo timeout a partir de $n=1536$ en Python y no en C o Java?"*
> **Tu respuesta:**  
> *"El protocolo experimental fijó un presupuesto de seguridad de 60.0 segundos por repetición. En $n=1024$, Python ya consumía 41.15 segundos. Al multiplicar la dimensión por 1.5 ($n=1536$), el volumen de operaciones cúbicas crece por $1.5^3 = 3.375$.  
> $41.15 \times 3.375 \approx 139$ segundos.  
> Como 139 segundos duplica con creces el límite de 60 segundos, el arnés de benchmark abortó la ejecución al cumplirse el minuto. En cambio, C optimizado en $n=1536$ tardó apenas 4.25 segundos y Java 5.34 segundos, por lo que pudieron llegar a $n=2048$ con total comodidad."*

---

### Pregunta 4: *"¿Por qué la memoria de Python es 12.8 veces mayor si las matrices solo ocupan 24 MB?"*
> **Tu respuesta:**  
> *"Por la arquitectura de datos de CPython. En C, un array de dobles ocupa estrictamente 8 bytes por elemento. En Python, una lista de listas de flotantes requiere:
> 1. Un objeto `PyFloatObject` por cada número (24 bytes: 8 de contador de referencias, 8 de puntero a tipo y 8 del valor double).
> 2. Un puntero de 8 bytes en la lista contenedora.  
> Esto da un mínimo teórico de 32 bytes por entrada (un factor de 4x, medido en ~74 MB con `tracemalloc`).  
> Y a nivel de proceso del sistema operativo, medido con `getrusage`, el Peak RSS llegó a 306 MB debido al mapeo de librerías del intérprete, a los bloques de memoria preasignados por el asignador interno `pymalloc` (arenas y pools) y a la fragmentación del heap."*

---

### Pregunta 5: *"¿Por qué no transpusiste la matriz B o usaste NumPy?"*
> **Tu respuesta:**  
> *"Porque el objetivo pedagógico y científico de este Assignment 1 era medir el **baseline canónico** estricto: el triple bucle $i\text{-}j\text{-}k$ sin optimizaciones algorítmicas, para estudiar de forma pura el coste del runtime y el impacto del hardware.  
> De hecho, usamos NumPy BLAS como verificación de corrección numérica (comprobando que el error relativo fuera menor a $10^{-12}$). Y precisamente el cuello de botella que hemos medido en el acceso por columnas de $B$ es la justificación motivacional perfecta para el Assignment 2, donde implementaremos la transposición, el bucle $i\text{-}k\text{-}j$ y el bloqueo de caché."*

---

## 4. Consejos de Expresión Oral y Puesta en Escena

1. **Uso de la herramienta web interactiva:**  
   Si presentas desde un portátil o navegador, abre `presentation/presentation.html`. Puedes pulsar la tecla `F` para ponerla en pantalla completa, activar o pausar el cronómetro con `T`, y tener el botón de notas `N` si necesitas consultar tus apuntes.
2. **Presentación con el PDF oficial:**  
   Si utilizas el proyector del aula con un visor de PDF (Acrobat, Preview), ponlo en modo presentación a pantalla completa (Cmd + L en Preview / Cmd + Shift + F en Acrobat).
3. **Control del ritmo:**  
   - Los primeros 30 segundos suelen ser los más rápidos por los nervios: respira hondo y habla con pausa en la Diapositiva 1.
   - En la Diapositiva 2, enfatiza el salto: *"41 segundos en Python frente a 1.2 segundos en C"*. Deja 2 segundos de silencio para que la audiencia asimile el contraste.
   - En la Diapositiva 3, cuando expliques que Java supera a C en $n=2048$, haz énfasis en que **el compilador JIT no es un juguete**, sino un optimizador de nivel de producción.
   - Termina en la Diapositiva 4 mirando al profesor y conectando con el siguiente tema de la asignatura (*Assignment 2*).
