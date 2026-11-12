// ==============================================================================
// GUÍA DE CONOCIMIENTO ESENCIAL: MULTIPLICACIÓN DE MATRICES (ASSIGNMENT 1)
// Documento de estudio y comprensión conceptual para la defensa oral
// Autor: Pablo Martínez Suárez · Asignatura: Big Data (GICD) · ULPGC
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
          Big Data · Guía de Fundamentos y Dominio Conceptual (Assignment 1)
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
        Universidad de Las Palmas de Gran Canaria · Grado en Ingeniería y Ciencia de Datos
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

// Paleta de colores temáticos
#let c-primary = rgb("#0284C7")     // Azul principal
#let c-dark = rgb("#0F172A")        // Texto oscuro
#let c-slate = rgb("#334155")       // Texto secundario
#let c-light-bg = rgb("#F8FAFC")    // Fondo neutro
#let c-border = rgb("#E2E8F0")      // Borde suave
#let c-green = rgb("#059669")       // Éxito / Java
#let c-green-bg = rgb("#ECFDF5")
#let c-amber = rgb("#D97706")       // Alerta / C -O0
#let c-amber-bg = rgb("#FFFBEB")
#let c-red = rgb("#DC2626")         // Crítico / Python
#let c-red-bg = rgb("#FEF2F2")
#let c-blue-bg = rgb("#F0F9FF")
#let c-purple = rgb("#7C3AED")      // Conceptos clave
#let c-purple-bg = rgb("#FAF5FF")

// Componentes visuales reutilizables
#let callout(title: none, icon: "💡", body, bg: c-light-bg, border: c-border, title-color: c-primary) = {
  block(
    width: 100%,
    fill: bg,
    stroke: (left: 3.5pt + title-color, rest: 0.75pt + border),
    radius: (right: 4pt, left: 0pt),
    inset: (x: 11pt, y: 8pt),
    breakable: false
  )[
    #if title != none [
      #block(below: 5pt)[
        #text(weight: "bold", size: 9.5pt, fill: title-color)[#icon #title]
      ]
    ]
    #body
  ]
}

#let concept-box(term, definition, analogy: none) = {
  block(
    width: 100%,
    fill: rgb("#FFFFFF"),
    stroke: 0.75pt + rgb("#CBD5E1"),
    radius: 4pt,
    inset: 9pt,
    breakable: false
  )[
    #text(weight: "bold", size: 10pt, fill: c-primary)[#term] \
    #v(2.5pt)
    #text(size: 9pt, fill: c-dark)[#definition]
    #if analogy != none [
      #v(3.5pt)
      #block(fill: c-blue-bg, inset: 5pt, radius: 3pt, width: 100%)[
        #text(size: 8.5pt, fill: rgb("#0369A1"), style: "italic")[*Metáfora sencilla:* #analogy]
      ]
    ]
  ]
}

#let stat-card(number, label, detail, color: c-primary) = {
  rect(
    width: 100%,
    fill: c-light-bg,
    stroke: 1pt + color,
    radius: 4pt,
    inset: 7pt
  )[
    #align(center)[
      #text(size: 15pt, weight: "bold", fill: color)[#number] \
      #v(-2pt)
      #text(size: 8.5pt, weight: "bold", fill: c-dark)[#label] \
      #v(-3pt)
      #text(size: 7.5pt, fill: rgb("#64748B"))[#detail]
    ]
  ]
}

// ------------------------------------------------------------------------------
// ENCABEZADO PRINCIPAL
// ------------------------------------------------------------------------------

#align(center)[
  #block(fill: c-blue-bg, inset: (x: 10pt, y: 4pt), radius: 10pt)[
    #text(size: 8pt, weight: "bold", fill: c-primary, tracking: 1pt)[
      MATERIAL DE DOMINIO CONCEPTUAL · BIG DATA (GICD) · CURSO 2026–2027
    ]
  ]
  #v(5pt)
  #text(size: 19pt, weight: "bold", fill: c-dark)[
    Guía de Conocimiento Esencial para la Exposición
  ] \
  #v(2pt)
  #text(size: 12pt, weight: "medium", fill: rgb("#0284C7"))[
    Multiplicación de Matrices en C, Java y Python: Fundamentos, Causas y Resultados
  ] \
  #v(4pt)
  #text(size: 9pt, fill: rgb("#64748B"))[
    *Pablo Martínez Suárez* · Departamento de Informática y Sistemas · ULPGC
  ]
]

#v(4pt)
#line(length: 100%, stroke: 1pt + c-primary)
#v(6pt)

#callout(title: "¿Cómo estudiar con este documento?", icon: "🎯", bg: c-purple-bg, border: rgb("#E9D5FF"), title-color: c-purple)[
  *Esta guía no es un guión teatral para memorizar palabra por palabra.* Aprenderse un texto de memoria te hace hablar con tensión y, si te trabas o te hacen una pregunta en medio, te quedas en blanco.

  Aquí encontrarás *el conocimiento detrás de cada número y cada gráfica*: qué hicimos exactamente, por qué ocurren las diferencias entre lenguajes y qué le pasa por dentro al chip del ordenador. Con esta comprensión clara podrás salir a defender tu trabajo con seguridad, naturalidad y tus propias palabras.
]

#v(6pt)

// ------------------------------------------------------------------------------
// SECCIÓN 1: EL MAPA MENTAL DEL TRABAJO
// ------------------------------------------------------------------------------
= El Mapa Mental: ¿Qué experimento hemos hecho y para qué?

Antes de entrar en tecnicismos, ten siempre presente la visión global:

#concept-box(
  "El Experimento en pocas palabras",
  [Hemos medido cuánto tiempo tarda y cuánta memoria gasta multiplicar dos matrices cuadradas de números decimales de 64 bits ($A times B = C$), usando el algoritmo canónico más básico (el triple bucle $i-j-k$), bajo condiciones estrictamente idénticas en tres paradigmas de programación: *C* (compilación nativa AOT), *Java* (máquina virtual OpenJDK 22 con JIT HotSpot C2) y *Python* (interpretación dinámica con CPython 3.14).],
  analogy: "Es como una carrera de atletismo de 100 metros lisos donde todos corren exactamente por la misma pista: uno corre con zapatillas profesionales de clavos (C), otro es un atleta que calienta y corre cada vez más rápido hasta alcanzar la velocidad máxima (Java JIT), y el tercero corre cargando maletas pesadas que tiene que abrir en cada zancada (Python)."
)

#v(4pt)

== ¿Por qué multiplicar matrices y por qué el algoritmo básico?
1. *El núcleo del Big Data:* Prácticamente todo el procesamiento masivo de datos y el Machine Learning moderno (redes profundas, LLMs, filtrado colaborativo, regresiones) consiste en billones de multiplicaciones matriciales.
2. *Objetivo pedagógico de la Práctica 1:* Queremos evaluar el rendimiento *baseline* (la línea de base sin optimizar). Antes de aplicar técnicas avanzadas en la Práctica 2, medimos cómo rinde cada lenguaje con el código ingenuo para evidenciar las ineficiencias del software y los límites de la memoria física.

== La regla matemática: Complejidad Cúbica $O(n^3)$
Para multiplicar dos matrices cuadradas de dimensión $n times n$:
- El resultado es una matriz de $n times n = n^2$ casillas.
- Para calcular cada casilla individual, se multiplican $n$ pares de números y se suman: $n$ multiplicaciones y $n$ adiciones (un producto escalar).
- Total de operaciones de coma flotante (FLOPs): $bold(2n^3)$.

#callout(title: "¿Qué significa el exponente cúbico en la práctica?", icon: "⚠️", bg: c-amber-bg, border: rgb("#FDE68A"), title-color: c-amber)[
  *Efecto de escala cúbica:* Si duplicas la dimensión de la matriz (por ejemplo, pasar de $n=512$ a $n=1024$), el volumen de trabajo matemático no se duplica, ¡se multiplica por $2^3 = bold(8)$! Si triplicas la dimensión, se multiplica por $3^3 = bold(27)$. Por eso el tiempo de cálculo se dispara verticalmente a poco que la matriz crezca.
]

#v(6pt)

// ------------------------------------------------------------------------------
// SECCIÓN 2: EL RIGOR EXPERIMENTAL (SLIDE 1)
// ------------------------------------------------------------------------------
= El Rigor Experimental: Las reglas del juego (Slide 1)

En ciencia e ingeniería, unas mediciones mal aisladas no sirven para nada. En la primera diapositiva dejas claro que tu experimento es reproducible y riguroso. Estos son los 5 conceptos que debes tener claros:

#grid(
  columns: (1fr, 1fr),
  gutter: 8pt,
  concept-box(
    "1. SplitMix64 determinista (Semilla 42)",
    [Generamos los datos de entrada con exactamente el mismo algoritmo en C, Java y Python, arrancando con la misma semilla ($S=42$).],
    analogy: "Asegura que los tres lenguajes multipliquen exactamente las mismas matrices bit a bit. Si uno gana, no es por haber tenido números más fáciles."
  ),
  concept-box(
    "2. Aislamiento del Cronómetro",
    [El cronómetro solo mide el triple bucle de cálculo (`for i`, `for j`, `for k`).],
    analogy: "No cronometramos reservar memoria ni mostrar texto por pantalla. Medimos pura y exclusivamente la velocidad de cálculo aritmético."
  )
)

#v(4pt)

#grid(
  columns: (1fr, 1fr),
  gutter: 8pt,
  concept-box(
    "3. Defensa contra Código Muerto (DCE)",
    [Los compiladores optimizadores pueden borrar el bucle entero si detectan que el resultado no se usa (dando tiempo cero artificial).],
    analogy: "Al terminar el cronómetro, calculamos un checksum (la suma de todos los elementos). Al obligar a usar el resultado, el compilador está obligado a calcularlo todo."
  ),
  concept-box(
    "4. Calentamiento (Warm-Up)",
    [Antes de medir oficialmente, ejecutamos entre 1 y 3 rondas previas de prueba que descartamos.],
    analogy: "La máquina virtual de Java arranca en frío interpretando código. Necesita unas vueltas previas para que su compilador inteligente (JIT) se active y estabilice su velocidad."
  )
)

#v(4pt)

#concept-box(
  "5. Mediana e IQR en lugar de la Media clásica",
  [Hicimos 5 repeticiones por cada configuración. En vez de promediar con la media, usamos la *mediana* (valor central ordenado) y el *IQR* (rango intercuartílico, la dispersión del 50% central).],
  analogy: "Si tu ordenador sufre una pausa puntual del sistema operativo o abre un proceso en segundo plano en una repetición, la media se distorsionaría. La mediana es robusta y no se deja engañar por datos atípicos."
)

#v(6pt)

// ------------------------------------------------------------------------------
// SECCIÓN 3: LOS TIEMPOS Y EL MURO DEL TIMEOUT (SLIDE 2)
// ------------------------------------------------------------------------------
= Tiempos de Ejecución y la Barrera de Viabilidad (Slide 2)

Aquí mostramos los datos reales medidos en el Apple M2. Guarda en la memoria estos números de referencia para $n=1024$:

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 6pt,
  stat-card("1.21 s", "C (Clang -O2)", "Baseline de máxima velocidad", color: c-primary),
  stat-card("1.76 s", "Java HotSpot", "Muy cerca de C (1.45x)", color: c-green),
  stat-card("3.80 s", "C sin optimizar (-O0)", "3.15x más lento que -O2", color: c-amber),
  stat-card("41.15 s", "Python 3.14", "34x más lento que C", color: c-red)
)

#v(6pt)

== ¿Por qué todos siguen la curva cúbica pero Python tarda 34 veces más?
Al graficar los datos en escala logarítmica, las curvas de los tres lenguajes son rectas paralelas con pendientes $alpha approx 3.10$ a $3.38$.
- *¿Qué confirma esto?* Que la matemática no miente: todos ejecutan una carga de trabajo de $O(n^3)$.
- *¿Dónde radica la diferencia?* En el *factor constante* ($c$) de la ecuación $T(n) = c dot n^alpha$.
  - C (`-O2`) tiene una constante $c = 1.01 times 10^(-10)$ segundos por operación.
  - Python tiene una constante $c = 1.91 times 10^(-8)$ segundos por operación.
  - *La constante de Python es 189 veces peor.* Es decir, para resolver *la misma operación matemática elemental*, Python invierte 189 veces más tiempo que C.

#v(4pt)

#callout(title: "El Hallazgo Crítico: El muro de los 60 segundos (Timeout en n = 1536)", icon: "🛑", bg: c-red-bg, border: rgb("#FECACA"), title-color: c-red)[
  *¿Qué le ocurrió a Python a partir de $n=1536$?*
  Fijamos por protocolo un límite de seguridad de 60.0 segundos por repetición.
  - En $n=1024$, Python tardó 41.15 segundos (dentro del presupuesto).
  - Al subir a $n=1536$, la dimensión crece un factor de $1.5$. Por la ley cúbica, el tiempo crece por $1.5^3 = bold(3.375)$.
  - Pronóstico para Python: $41.15 times 3.375 approx bold(139 text(" segundos"))$ (más de 2 minutos).
  - El arnés de benchmark abortó la ejecución al superar el minuto (*TIMEOUT*).
  - *Conclusión:* El límite operativo de Python puro en esta máquina es estrictamente $bold(n=1024)$. En cambio, C y Java llegaron sin problemas hasta $n=2048$ (tardando 29s y 19s).
]

#v(6pt)

// ------------------------------------------------------------------------------
// SECCIÓN 4: BAJO EL CAPÓ DE CADA LENGUAJE (SLIDE 3)
// ------------------------------------------------------------------------------
= Arquitectura Interna: ¿Por qué rinde así cada lenguaje? (Slide 3)

Esta es la clave para destacar en la presentación: explicar el *motivo físico de hardware* por el que los números son como son.

#v(3pt)

== 1. El cuello de botella de Python (Techo de 0.06 GFLOPS)
Python no supera los 0.06 GFLOPS (menos del 1.5% de la velocidad de C). Esto ocurre por dos motivos:
#grid(
  columns: (1fr, 1fr),
  gutter: 8pt,
  concept-box(
    "A. El bucle del intérprete (ceval.c)",
    [Python no entrega instrucciones directas a la CPU. En cada iteración del bucle más interno, el intérprete en C (`ceval.c`) tiene que decodificar bytecodes abstractos (`LOAD_FAST`, `BINARY_OP`), comprobar dinámicamente si los tipos son flotantes y verificar errores.],
    analogy: "Es como un traductor que traduce cada palabra en tiempo real buscando en un diccionario, en lugar de un texto ya traducido y listo para leer."
  ),
  concept-box(
    "B. El 'Boxing' de Objetos (PyFloatObject)",
    [En C, un número decimal de 64 bits son 8 bytes contiguos en memoria. En Python, cada número es un objeto completo en el montón (*heap*) que ocupa 24 bytes, más un puntero de 8 bytes en la lista (total: 32 bytes).],
    analogy: "Para hacer una multiplicación, la CPU debe seguir dos saltos de punteros en memoria para desempaquetar el valor real. La CPU se pasa el tiempo esperando a la memoria en vez de calculando."
  )
)

#v(4pt)

== 2. C con `-O2` vs C con `-O0`: El valor del compilador
- *Con `-O0` (sin optimizar, 3.80s en $n=1024$):* El compilador Clang genera código ingenuo. En cada iteración del bucle $k$, guarda el acumulador `sum` en la memoria RAM (en la pila o *stack*) y lo vuelve a leer. Esas lecturas y escrituras constantes en memoria penalizan el tiempo un factor de *3.15x*.
- *Con `-O2` (optimizado, 1.21s en $n=1024$):* El compilador retiene `sum` directamente en un *registro físico de la CPU* (el registro de coma flotante `d0` de ARM64), que se lee en 0 ciclos de espera. Además, desenrolla bucles y reordena instrucciones para mantener ocupadas las unidades aritméticas.
- *Dato revelador:* Incluso C sin optimizar (`-O0`) es *10.8 veces más rápido que Python*. Esto prueba que la ejecución nativa sin intérprete y sin objetos dinámicos aporta una ventaja basal abrumadora.

#v(4pt)

== 3. Java HotSpot C2: El motor JIT que alcanza a C
El resultado más llamativo del estudio:
- En $n=1024$, Java tarda *1.76 segundos* (muy cerca de los 1.21s de C).
- Y en $n=2048$, Java tarda *19.1 segundos* y *supera* a C optimizado (*29.0 s*).

#callout(title: "¿Cómo funciona la Máquina Virtual HotSpot de Java?", icon: "⚡", bg: c-green-bg, border: rgb("#A7F3D0"), title-color: c-green)[
  1. *Compilación escalonada (Tiered Compilation):* Java arranca interpretando bytecode. Unos contadores internos identifican las regiones que se ejecutan millones de veces (los *hot spots*).
  2. *El compilador C2 (Tier 4):* Al detectar el bucle de multiplicación, el compilador C2 traduce el código en tiempo de ejecución a código máquina nativo del Apple M2.
  3. *Vectorización ARM64 NEON:* C2 desenrolla el bucle y genera instrucciones vectoriales *FMA (Fused Multiply-Accumulate)* de 128 bits, realizando la multiplicación y la suma en un solo ciclo.
  4. *Por qué supera a C en $n=2048$:* Clang con `-O2` no vectorizó agresivamente el bucle interno para respetar estrictamente las reglas de redondeo flotante de C, mientras que C2 generó una secuencia de instrucciones vectoriales muy eficiente. Además, la estructura `double[][]` de Java fragmentó las filas en memoria, dispersando mejor las líneas de caché.
  5. *El coste del calentamiento:* Si medimos Java en frío (primera iteración sin warm-up), tarda hasta *3.5 veces más* por la sobrecarga inicial de compilación y monitorización.
]

#v(6pt)

// ------------------------------------------------------------------------------
// SECCIÓN 5: MEMORIA Y EL MURO DE CACHÉ (SLIDE 4)
// ------------------------------------------------------------------------------
= Memoria y el Gran Muro de Caché (Slide 4)

En ingeniería de Big Data, la memoria casi siempre es el auténtico factor limitante:

== 1. La inflación de memoria de Python (306 MB vs 24 MB teóricos)
Tres matrices cuadradas de $1024 times 1024$ números double de 8 bytes ocupan matemáticamente:
$ 3 times 1024^2 times 8 text(" bytes") = bold(24.0 text(" MiB (25.17 MB decimales)")) $
- *C:* El consumo del proceso (Peak RSS) fue de *25.3 MB* ($1.05 times$ la carga teórica).
- *Java:* El heap ocupado por las matrices fue de *23.8 MB* (las matrices puras).
- *Python:* El proceso consumió ¡*306.3 MB*! ($bold(12.8 times)$ más que los datos reales).
  - *Causas:* El boxing de 32 bytes por número ($4 times$), la fragmentación del montón y los bloques preasignados en memoria (*arenas y pools*) que mantiene el gestor `pymalloc` de Python.

#v(4pt)

== 2. El Muro de Caché L2: El drama del acceso por columnas
¿Por qué en $n=2048$ C tardó 29.0s frente a 1.2s en $n=1024$? El trabajo creció 8 veces ($2^3$), pero el tiempo creció *24 veces*.

#concept-box(
  "El cuello de botella del algoritmo i-j-k: Stride de la Matriz B",
  [En la memoria del ordenador, las matrices se almacenan por filas continuas (*row-major*):
  - La matriz $A$ se recorre por filas: $A[i][k]$. Cada dato está contiguo al anterior en memoria $arrow$ *Localidad espacial perfecta*.
  - ¡Pero la matriz $B$ se recorre por columnas: $B[k][j]$! En cada vuelta de $k$, saltar al siguiente elemento de la misma columna exige saltar $8n$ bytes hacia adelante en la memoria RAM.],
  analogy: "Leer la matriz A es como leer un libro línea por línea. Leer la matriz B es como leer la primera palabra de la página 1, saltar a la primera de la página 2, luego a la de la página 3... Te pasas la vida pasando páginas."
)

#v(4pt)

#callout(title: "¿Por qué se desborda la caché del Apple M2?", icon: "🧱", bg: c-blue-bg, border: rgb("#BAE6FD"), title-color: c-primary)[
  - Tu procesador Apple M2 tiene *16 MB de memoria caché L2* compartida entre los núcleos.
  - A $n=1024$, una matriz ocupa 8 MB, por lo que cabe cómodamente dentro de la caché.
  - A $n=2048$, ¡una sola matriz ocupa *32 MB*!
  - Al recorrer $B$ a saltos por columnas, cada acceso genera un *fallo de caché (cache miss)*. La CPU se queda congelada esperando a que la memoria RAM convencional (LPDDR5) le suministre los datos. El rendimiento se degrada a velocidad de bus de memoria.
]

#v(6pt)

// ------------------------------------------------------------------------------
// SECCIÓN 6: LAS 4 CONCLUSIONES CLAVE
// ------------------------------------------------------------------------------
= Las 4 Conclusiones que debes transmitir

Estas son las cuatro lecciones que cierran tu exposición con solvencia profesional:

#rect(width: 100%, stroke: 0.75pt + c-border, fill: c-light-bg, radius: 4pt, inset: 9pt)[
  #grid(
    columns: (auto, 1fr),
    gutter: 10pt,
    [#circle(radius: 10pt, fill: c-red)[#align(center + horizon)[#text(fill: white, weight: "bold", size: 8pt)[1]]]],
    [
      *Nunca uses bucles en Python puro para cálculo numérico intensivo:*  
      En Big Data usamos la sintaxis cómoda de Python, pero el motor debe ser una librería compilada en C/C++ o Fortran (como NumPy, PyTorch o Polars con BLAS subyacente).
    ]
  )
  #v(4pt)
  #line(length: 100%, stroke: 0.5pt + c-border)
  #v(4pt)
  #grid(
    columns: (auto, 1fr),
    gutter: 10pt,
    [#circle(radius: 10pt, fill: c-green)[#align(center + horizon)[#text(fill: white, weight: "bold", size: 8pt)[2]]]],
    [
      *Java es un entorno plenamente competitivo para Big Data:*  
      Con la máquina virtual estabilizada tras warm-up, HotSpot C2 rinde a la par que C compilado. Por eso sistemas distribuidos líderes como *Apache Spark, Hadoop, Flink y Kafka* se ejecutan en la JVM.
    ]
  )
  #v(4pt)
  #line(length: 100%, stroke: 0.5pt + c-border)
  #v(4pt)
  #grid(
    columns: (auto, 1fr),
    gutter: 10pt,
    [#circle(radius: 10pt, fill: c-primary)[#align(center + horizon)[#text(fill: white, weight: "bold", size: 8pt)[3]]]],
    [
      *La física del hardware prima sobre la complejidad teórica:*  
      Aunque la fórmula sea $O(n^3)$ en todos los lenguajes, si el patrón de acceso a memoria rompe la jerarquía de cachés, los fallos de acceso a RAM arruinan el rendimiento.
    ]
  )
  #v(4pt)
  #line(length: 100%, stroke: 0.5pt + c-border)
  #v(4pt)
  #grid(
    columns: (auto, 1fr),
    gutter: 10pt,
    [#circle(radius: 10pt, fill: c-purple)[#align(center + horizon)[#text(fill: white, weight: "bold", size: 8pt)[4]]]],
    [
      *El puente motivacional hacia la Práctica 2 (Assignment 2):*  
      ¿Cómo resolvemos el cuello de botella de la matriz $B$? Reordenando los bucles a $bold(i-k-j)$ (para recorrer ambas matrices por filas contiguas) y aplicando *cache tiling* (partir la matriz en bloques que quepan en la caché L1/L2).
    ]
  )
]

#v(6pt)

// ------------------------------------------------------------------------------
// SECCIÓN 7: VISTA DE PÁJARO DE TUS 4 DIAPOSITIVAS
// ------------------------------------------------------------------------------
= Estructura Conceptual de tus 4 Diapositivas

Cuando mires las diapositivas en la pantalla, este es el mensaje que debes transmitir en cada una:

#table(
  columns: (auto, 1.1fr, 2.2fr),
  align: (center, left, left),
  stroke: 0.5pt + c-border,
  fill: (col, row) => if row == 0 { c-blue-bg } else { none },
  table.header([*Diapositiva*], [*Tema Central*], [*Idea conceptual que debes explicar*]),
  [Slide 1],
  [*Rigor Experimental* \ (Protocolo y Testbed)],
  [Condiciones idénticas: Apple M2 monohilo, generador SplitMix64 idéntico con semilla 42, cronómetro que aísla solo el bucle aritmético, calentamiento previo y defensa contra código muerto con checksum.],
  
  [Slide 2],
  [*Escalado Temporal* \ (Tiempos y Timeout)],
  [Todos cumplen la ley cúbica ($O(n^3)$), pero la constante de Python es 189 veces mayor. A $n=1024$ Python tarda 41s, y a $n=1536$ choca contra el timeout de 60s (proyectado ~139s). Su límite es $n=1024$.],
  
  [Slide 3],
  [*Análisis Arquitectónico* \ (Throughput y JIT)],
  [¿Por qué ocurre? Python se estanca en 0.06 GFLOPS (intérprete + boxing), C optimizado supera a `-O0` en 3.15x (registros vs RAM stack), y Java HotSpot C2 alcanza a C gracias a vectorización NEON FMA.],
  
  [Slide 4],
  [*Memoria y Caché* \ (Conclusiones Big Data)],
  [Python infla la memoria 13 veces (306 MB vs 24 MB teóricos). El salto por columnas de la matriz B desborda la caché L2 de 16 MB a $n=2048$. Cerrar con las 4 conclusiones y el puente a la Práctica 2.]
)

#v(6pt)

// ------------------------------------------------------------------------------
// SECCIÓN 8: BANCO DE PREGUNTAS TÍPICAS (Q&A DE DEFENSA)
// ------------------------------------------------------------------------------
= Banco de Preguntas del Evaluador: Respuestas con sentido común

Si el profesor o los evaluadores te preguntan al final, estas son las explicaciones lógicas:

#concept-box(
  "Pregunta 1: '¿Por qué en n=2048 Java tardó 19 segundos y C tardó 29 segundos si C es nativo?'",
  [*Respuesta conceptual:*
  'Ambos códigos implementan el bucle básico sin alterar el orden. Sin embargo, en Clang `-O2`, el compilador estático no reordena las operaciones flotantes para no alterar la asociatividad matemática de IEEE 754.  
  En cambio, el compilador JIT C2 de Java perfiló el código en ejecución sobre el chip M2 y aplicó un desenrollado agresivo del bucle junto con instrucciones vectoriales ARM64 NEON FMA (Fused Multiply-Accumulate). Además, la representación en Java de array de arrays (`double[][]`) fragmentó ligeramente las filas en memoria, lo que paradójicamente mitigó los conflictos de líneas de caché frente al array plano continuo de C cuando la matriz superó el tamaño de la caché L2.'],
  analogy: "C preparó la estrategia antes de empezar la carrera sin ver el circuito; Java observó las primeras vueltas y optimizó las ruedas en vivo para la pista del procesador."
)

#v(4pt)

#concept-box(
  "Pregunta 2: '¿Por qué la memoria de Java en la tabla lleva una cruz (†)?'",
  [*Respuesta conceptual:*
  'Porque son métricas distintas requeridas por el guión de la práctica. En C y Python medimos el tamaño total del proceso en el sistema operativo (*Peak RSS* con `getrusage`), que incluye las librerías del ejecutable y todo el espacio reservado.  
  En Java medimos la variación de memoria administrada dentro del heap con `Runtime.getRuntime()`. Si hubiéramos medido el proceso completo de Java en el sistema operativo, se sumarían unos 80 MB adicionales correspondientes a la propia infraestructura de la Máquina Virtual (metaspace, búferes de compilación JIT y tablas del recolector de basura).'],
  analogy: "Es como comparar el peso de una maleta sola (heap de matrices) frente al peso de una persona cargando la maleta (proceso completo en el SO)."
)

#v(4pt)

#concept-box(
  "Pregunta 3: '¿Por qué no usaste NumPy si en la vida real nadie multiplica matrices en Python puro?'",
  [*Respuesta conceptual:*
  'Porque el objetivo del experimento era evaluar el coste intrínseco de cada lenguaje con el algoritmo canónico. De hecho, usamos NumPy BLAS como patrón de referencia (*ground truth*) para validar que el resultado numérico fuera correcto con un error menor a $10^(-12)$. Si hubiéramos medido NumPy en los gráficos temporales, habríamos estado evaluando una librería de álgebra lineal escrita en C y Fortran, no la velocidad del intérprete de Python.'],
  analogy: "Queríamos medir cuánto corre un humano a pie. Ponerle NumPy es subirlo a una moto: ya no estás evaluando al atleta, estás evaluando el motor de la moto."
)

#v(4pt)

#concept-box(
  "Pregunta 4: '¿Por qué el exponente alfa de C con -O2 dio 3.38 en vez del 3.00 teórico?'",
  [*Respuesta conceptual:*
  'El 3.00 teórico asume que cada operación aritmética tarda siempre el mismo tiempo exacto. Pero en un procesador real, cuando la matriz supera los 1024 elementos, los datos ya no caben en la memoria caché rápida (16 MB de L2) y el chip tiene que esperar a la memoria RAM, que tiene una latencia mucho mayor. Ese sobrecoste acumulado en los tamaños más grandes curva la gráfica hacia arriba, elevando el exponente empírico de 3.00 a 3.38.'],
  analogy: "Si tienes tus herramientas en la mesa tardas 2 segundos por tornillo. Si la mesa se llena y tienes que ir al garaje a buscar cada tornillo, el tiempo medio por tornillo se dispara."
)

#v(8pt)
#align(center)[
  #text(size: 8.5pt, fill: rgb("#94A3B8"), style: "italic")[
    Guía de Conocimiento Esencial · Pablo Martínez Suárez · Práctica 1 de Big Data · ULPGC
  ]
]
