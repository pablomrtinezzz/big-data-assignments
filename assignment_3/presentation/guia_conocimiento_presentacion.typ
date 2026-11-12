// ==============================================================================
// GUÍA DE CONOCIMIENTO Y DEFENSA ORAL: ASSIGNMENT 3
// Paralelización en Memoria Compartida y Vectorización SIMD
// Autor: Pablo Martínez Suárez · Asignatura: Big Data (GCID) · ULPGC (2026–2027)
// ==============================================================================

#set page(
  paper: "a4",
  margin: (x: 1.8cm, top: 2.0cm, bottom: 2.0cm),
  header: context [
    #if counter(page).get().first() > 1 [
      #grid(
        columns: (1fr, auto),
        align: (left, right),
        text(size: 8pt, fill: rgb("#64748B"), weight: "medium")[
          Big Data · Guía de Dominio Conceptual y Defensa Oral (Assignment 3)
        ],
        text(size: 8pt, fill: rgb("#0284C7"), weight: "bold")[
          Pablo Martínez Suárez
        ]
      )
      #v(2pt)
      #line(length: 100%, stroke: 0.5pt + rgb("#CBD5E1"))
    ]
  ],
  footer: context [
    #line(length: 100%, stroke: 0.5pt + rgb("#E2E8F0"))
    #v(3pt)
    #grid(
      columns: (1fr, auto),
      text(size: 8pt, fill: rgb("#94A3B8"))[
        Universidad de Las Palmas de Gran Canaria · Grado en Ciencia e Ingeniería de Datos
      ],
      text(size: 8pt, fill: rgb("#64748B"), weight: "bold")[
        Página #counter(page).display("1 de 1", both: true)
      ]
    )
  ]
)

#set text(
  font: ("Helvetica Neue", "Arial"),
  size: 9.5pt,
  fill: rgb("#0F172A"),
  lang: "es"
)

#set par(justify: true, leading: 0.58em)
#set heading(numbering: "1.1")

#align(center)[
  #v(0.5em)
  #text(size: 16pt, weight: "bold", fill: rgb("#0369A1"))[
    Guía Maestra de Defensa Oral y Dominio Conceptual
  ] \
  #v(0.3em)
  #text(size: 12pt, weight: "medium", fill: rgb("#334155"))[
    Práctica 3: Paralelización en Memoria Compartida y Vectorización ARM NEON
  ] \
  #v(0.3em)
  #text(size: 9pt, fill: rgb("#64748B"))[
    *Pablo Martínez Suárez* · Big Data (Curso 2026–2027) · GCID · ULPGC \
    Repositorio: #link("https://github.com/pablomrtinezzz/big-data-assignments")
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.0pt + rgb("#0284C7"))
]

#v(0.5em)

== 1. Estructura Temporal y Presupuesto de Tiempo (Time Budget)

La defensa oral está estructurada estrictamente para completarse en *3 minutos y 30 segundos*, permitiendo una reserva de más de un minuto para responder preguntas del tribunal sin alcanzar el tope de 5 minutos.

#table(
  columns: (auto, 1fr, auto, auto),
  align: (center, left, center, center),
  stroke: (x, y) => if y == 0 { (bottom: 1.2pt + rgb("#0284C7")) } else { (bottom: 0.4pt + rgb("#E2E8F0")) },
  table.header([*Diapositiva*], [*Núcleo Temático*], [*Tiempo Sugerido*], [*Minuto Acumulado*]),
  [Slide 1], [Rigor Experimental, Arquitectura Apple M2 (4P + 4E), PRNG y Ausencia de Data Races], [45 s], [0:00 – 0:45],
  [Slide 2], [Escalado Fuerte, Overhead en $p=1$, Ley de Amdahl y Regresión por Sobreasignación], [60 s], [0:45 – 1:45],
  [Slide 3], [Impacto de la Heterogeneidad de Núcleos y Planificación Dinámica (+11.7% GFLOPS)], [65 s], [1:45 – 2:50],
  [Slide 4], [Vectorización ARM NEON (`fmla.2d`), Muro de Memoria L2 y Puente hacia la Práctica 4], [55 s], [2:50 – 3:45],
  [Defensa], [Turno de Preguntas y Respuestas con el Profesor / Tribunal], [~1 min], [3:45 – 5:00],
)

== 2. Guión Detallado para la Exposición Oral

=== Diapositiva 1: Rigor Experimental y Protocolo Metodológico
- *Mensaje Central:* "Evaluamos el paralelismo en memoria compartida de forma rigurosa y reproducible sobre un SoC asimétrico real (Apple M2), garantizando bit a bit la misma carga aritmética y ausencia absoluta de data races."
- *Guión Hablado:*
  _"Buenos días. En esta tercera práctica investigamos el impacto del número de hilos $p$ (desde 1 hasta 16) sobre el rendimiento, el factor de aceleración (speedup) y la eficiencia del producto de matrices denso sobre una plataforma multi-núcleo asimétrica: el procesador Apple M2, integrado por 4 núcleos de alto rendimiento Avalanche (3.5 GHz) y 4 núcleos de alta eficiencia Blizzard (2.42 GHz)._

  _Siguiendo el protocolo común de la asignatura, el algoritmo evaluado es el bucle reordenado $i-k-j$, que maximiza la localidad espacial con accesos contiguos en memoria tanto a la matriz $B$ como a $C$. La versión paralela se compara estrictamente frente a una línea base secuencial equivalente compilada sin el runtime de OpenMP. Cada hilo es dueño exclusivo de la fila $i$ del resultado, asegurando cero colisiones de escritura sin necesidad de cerrojos atómicos. Las entradas se inicializan deterministamente con el PRNG SplitMix64 (semilla 42), el temporizador monótono aísla solo el cómputo y se reportan mediana e IQR de 5 repeticiones tras calentamiento."_

=== Diapositiva 2: Escalado Fuerte y Regímenes Microarquitectónicos
- *Mensaje Central:* "El escalado empírico confirma tres fases diferenciadas: la penalización de framework en $p=1$, la aceleración en los núcleos de rendimiento, y el límite por asimetría y sobreasignación."
- *Guión Hablado:*
  _"Al observar las curvas de escalado fuerte, detectamos tres regímenes fundamentales:_

  _En primer lugar, con un solo hilo ($p=1$), OpenMP introduce una penalización del 47% al 53% respecto al secuencial puro. Esta medición solicitada en la guía docente evidencia el coste de instanciación del pool de hilos y la sobrecarga de barreras en el runtime._

  _En segundo lugar, en el Régimen I ($p=2$ y $p=4$), la ejecución se confina a los núcleos de alto rendimiento, escalando limpiamente hasta 22.8 GFLOPS en $n=1024$. En el Régimen II ($p=6$ a $8$), se incorporan los núcleos de eficiencia; al poseer menor IPC y frecuencia reducida (2.42 GHz), el incremento marginal decrece. Finalmente, en el Régimen III ($p > 8$), la sobreasignación de hilos fuerza cambios de contexto involuntarios en el kernel, provocando que en $n=2048$ el tiempo empeore de 0.65 s a 0.80 s."_

=== Diapositiva 3: Estrategias de Planificación y Heterogeneidad de Núcleos
- *Mensaje Central:* "El reparto estático de trabajo colapsa en procesadores heterogéneos; el reparto dinámico aprovecha la disparidad de potencia de los núcleos y eleva el throughput un +11.7%."
- *Guión Hablado:*
  _"El hallazgo metodológico clave de esta práctica reside en la interacción entre la política de planificación de OpenMP y la arquitectura heterogénea Big.LITTLE._

  _Con planificación estática (`schedule(static)`), el espacio de filas se divide en fragmentos iguales ($n/p$). Dado que los núcleos Avalanche son mucho más rápidos, finalizan su porción en la mitad de tiempo y quedan bloqueados en la barrera implícita esperando a los núcleos Blizzard lentos._

  _Al aplicar planificación dinámica (`schedule(dynamic, 8)`), los hilos solicitan lotes de 8 filas según van quedando desocupados. Esto permite que los núcleos de alto rendimiento absorban cerca del 70% del cómputo total. Como resultado empírico directo en $n=1024$ con 8 hilos, el rendimiento salta de 29.6 a 33.1 GFLOPS, logrando un incremento de velocidad del +11.7% y un speedup de 2.51x frente al C secuencial puro."_

=== Diapositiva 4: Vectorización ARM NEON y Limitaciones de Ancho de Banda
- *Mensaje Central:* "La aceleración vectorial mediante instrucciones NEON (`fmla.2d`) complementa el paralelismo a nivel de hilos, pero ambos chocan contra el muro de ancho de banda de memoria compartida."
- *Guión Hablado:*
  _"En la extensión opcional evaluamos la vectorización SIMD explícita utilizando los registros NEON de 128 bits de ARMv8. Demostramos la presencia de vectorización mediante diagnósticos de Clang y comprobando en el código ensamblador la emisión de instrucciones `fmla.2d`, capaces de procesar dos operaciones FMA de 64 bits por ciclo._

  _Sin embargo, a medida que la matriz escala ($n=2048$, donde las tres matrices ocupan 96 MB y superan con holgura los 16 MB de caché L2 compartida), los 8 núcleos compiten ferozmente por el bus de memoria unificada de 100 GB/s. Este fenómeno, denominado Muro de Memoria, estabiliza el rendimiento global._

  _Concluimos que la memoria compartida ofrece enormes ventajas en un único nodo, pero tiene un techo físico insalvable; para abordar problemas masivos de Big Data, la única vía viable es la distribución espacial de matrices entre nodos independientes (MPI o Apache Spark), objeto central de la Práctica 4. Muchas gracias."_

== 3. Banco de Preguntas Críticas para la Defensa (Q&A)

*P1: ¿Por qué OpenMP con $p=1$ es más lento que el ejecutable secuencial puro?* \
_Respuesta:_ La biblioteca `libomp` inicializa estructuras de control de concurrencia y reemplaza el bucle canónico por llamadas indirectas de despacho de trabajo en el runtime. Asimismo, en el ejecutable secuencial sin flags de threading, el compilador Clang dispone de mayor libertad para aplicar optimizaciones interprocedimentales sobre registros que se inhiben en presencia de bloques paralelos.

*P2: ¿Por qué en $p=16$ se degrada el rendimiento?* \
_Respuesta:_ El Apple M2 tiene exactamente 8 núcleos físicos de ejecución y no implementa Hyper-Threading (SMT). Levantar 16 hilos causa sobreasignación (oversubscription), obligando al planificador del sistema operativo a alternar hilos mediante cambios de contexto preventivos (preemptive context switches), lo que expulsa datos de las cachés L1D y L2 y añade latencia de sincronización en las barreras.

*P3: ¿Cómo se verificó que la vectorización realmente ocurre?* \
_Respuesta:_ Mediante dos evidencias objetivas: primera, los informes de pase de optimización del compilador (`-Rpass=loop-vectorize`) que confirman vectorización con factor 2; y segunda, la inspección del ensamblador (`clang -S`), donde se verifica la emisión directa de la instrucción vectorial `fmla.2d` en los registros vectoriales `v0-v31`, frente a la instrucción escalar `fmadd` en registros dobles `d0-d31`.

*P4: ¿Por qué no hay data races ni necesidad de secciones críticas en la matriz $C$?* \
_Respuesta:_ Porque la paralelización se efectúa sobre el bucle exterior $i$. Cada fila $i$ de la matriz $C$ pertenece exclusivamente a un hilo trabajador. Ningún hilo escribe en las posiciones de otro, y las matrices $A$ y $B$ se acceden en modo estrictamente de solo lectura, impidiendo colisiones y evitando false sharing dado que cada fila ocupa múltiplos enteros de la línea de caché de 64 bytes.
