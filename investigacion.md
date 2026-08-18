# Reporte de inteligencia: evaluación automatizada de lectura oral y fluidez lectora (2026-08-12)

> **Fecha de investigación:** 12 de agosto de 2026
> **Fuentes utilizadas:** Tavily (search advanced), DuckDuckGo, arXiv, y sitios oficiales de los proveedores
> **Tono:** informativo, verificable, sin recomendaciones de compra

---

## 0. Cómo leer este reporte

Este documento aplica las siguientes convenciones editoriales:

- distingue entre **prioritario Chile**, **complementario Chile**, **comparado internacional**, **benchmark regional**, **mercado/marco** y **referencia indirecta**;
- cuando una herramienta se conoce con varios nombres, se privilegia un nombre canónico y se listan alias;
- los recursos marcados como `Comparado`, `Benchmark regional` o `Mercado / marco` **no equivalen** a recomendación de adopción;
- la tabla maestra de este reporte consolida los recursos nuevos en formato reutilizable.

---

## 1. Resumen ejecutivo — 10 hallazgos que cambian el panorama

1. **Apareció un competidor chileno directo con canal MINEDUC.** `PerroLeo` (perroleo.cl) cierra el espacio identificado en abril como "la capa menos madura, pero con mayor potencial de innovación, en evaluación automatizada de lectura oral". PerroLeo opera con una **alianza explícita con Educarchile** ( Fundación Chile + MINEDUC), entrega **licencias gratuitas a establecimientos seleccionados** durante el segundo semestre de 2026 para 2° y 4° básico, declara cumplimiento de la **Ley 21.719** y una **rúbrica validada** para 1° a 8° básico.
2. **Un segundo competidor chileno ya está en aula.** `Leo Mejor` (leomejor.ai / leomejor.cl) fue lanzado a mediados de 2025 por **Imactiva SpA**, empresa con 16 a 20+ años de trayectoria, **patrocinio del MINEDUC**, presencia en **más de 4.000 colegios** y catálogo en **Por un Chile que Lee**. Reporta **+40 colegios** y **+10.000 mediciones** realizadas.
3. **El Estado chileno entra al juego.** La **Agencia de Calidad** implementará en 2026 la evaluación **Impulso Lector**, primera evaluación nacional que mide conjuntamente **comprensión lectora y fluidez lectora**. Esto mueve el piso del mercado: la fluidez deja de ser métrica opcional y pasa a ser estándar evaluativo nacional.
4. **La escala global probada supera cualquier proyección previa.** `Wadhwani AI ORF` (India) reporta **más de 15 millones de evaluaciones** (cifra enero 2026), **270.000 docentes** capacitados y **96.000 escuelas** asociadas. Es la prueba de escala más contundente de que ASR para ORF es viable en sistemas públicos masivos.
5. **La validación científica se consolidó.** El estudio de `van der Velde et al. 2025` (International Journal of Artificial Intelligence in Education, 653 niños holandeses, 176 h de audio) valida empíricamente el ASR como instrumento de medición ORF para lenguas semi-transparentes — familia tipológica del español. El estudio brasileño de `Viola et al. 2025` (Scientific Reports) confirma validación en portugués.
6. **El estado del arte técnico se mueve del ASR de palabras al MDD fonológico.** Los papers recientes de arXiv (`2604.22133`, `2606.22022`, `2511.20107`, `2506.12067`, `2311.07037`) muestran convergencia hacia detección de mispronunciación a nivel **fonémico y fonológico**, no ya palabra. `LUCA.ai` ya ofrece esto en producto (763.000+ mapeos grafema-fonema).
7. **SoapBox Labs fue adquirida por Curriculum Associates** y su tecnología se integró en **i-Ready**. Esto consolida la infraestructura ASR infantil dentro del canal comercial más grande de EE. UU. para lectura temprana.
8. **Amira se fusionó con Istation (junio 2024).** Operan hoy como "Intelligent Growth Engine" con evidencia ESSA Tier 2 (d = 0,06-0,70). Es uno de los pocos productos con ASR entrenado en **más de 10 mil millones de palabras infantiles**.
9. **El Sur Global dejó de ser sólo objeto de estudio.** Proyectos como **EGRA-AI** (consorcio Neurabuild + University of Cape Town + Binding Constraints Lab + Western Sydney University) y el caso **Bambara** (`arXiv:2606.31508`) publican benchmarks abiertos y apps de aula para lenguas africanas, replicando el modelo LMIC de Wadhwani.

---

## 2. Contexto: por qué esta actualización es necesaria

En los últimos meses han cambiado tres cosas en el panorama de la evaluación automatizada de la lectura oral:

1. **Aparecieron dos productos chilenos directos** en el nicho de evaluación automatizada de fluidez (PerroLeo, Leo Mejor), uno de ellos con el mismo canal institucional (Educarchile) que el MINEDUC usa para desplegar herramientas a escala nacional.
2. **El MINEDUC / Agencia de Calidad anunció Impulso Lector** como evaluación nacional, lo que institucionaliza la fluidez como métrica obligatoria del sistema y genera demanda explícita de herramientas de medición.
3. **La literatura científica del último año** (van der Velde 2025, Viola 2025, Frontiers 2026, y el bloque arXiv 2024-2026) subió el piso de evidencia esperable para cualquier herramienta nueva del sector.

Este reporte cubre esos tres frentes: ecosistema chileno, ecosistema internacional con novedades, y literatura académica aplicable.

---

## 3. Ecosistema chileno — fichas profundas

### 3.1 `PerroLeo` — el competidor chileno directo con canal MINEDUC

| Atributo | Valor |
|---|---|
| Nombre canónico | PerroLeo |
| Alias / URLs | perroleo.cl; "Perro Leo" |
| Origen / responsables | Equipo PerroLeo en alianza con Educarchile (Fundación Chile + MINEDUC) |
| Público objetivo | Estudiantes y docentes de 1° a 8° básico |
| Modelo de captura | El estudiante lee un texto en voz alta desde celular, tablet o computador; la plataforma analiza y entrega un informe "al instante" |
| Métricas declaradas | Velocidad lectora, calidad de la lectura oral, nivel de logro alcanzado comparado con lo esperado para el curso |
| Rúbrica | "rúbrica validada" (declarada en el sitio; sin detalle público del instrumento subyacente en la página de inicio) |
| Cobertura piloto 2026 | Licencias gratuitas a establecimientos seleccionados para **2° y 4° básico**, segundo semestre 2026, vía formulario de postulación evaluado por comité PerroLeo + Educarchile entre el 10 y 12 de agosto de 2026 |
| Marco de privacidad | Información regulada bajo **Ley 21.719** de protección de datos personales (citada explícitamente en la convocatoria) |
| Canal de despliegue | Educarchile (portal con convenio Fundación Chile + MINEDUC) — canal institucional consolidado |
| Modelo comercial | Licencias (gratuitas en convocatoria 2026; modelo de pago posterior no detallado públicamente) |
| Conectividad | Web / multi-dispositivo; requiere conexión para enviar audio y recibir informe |
| Evidencia de impacto | No hay estudio público de impacto hasta la fecha; la convocatoria 2026 es la primera ventana masiva de uso |
| Fecha de verificación | 2026-08-12 |

### 3.2 `Leo Mejor` — el competidor chileno ya en aula

| Atributo | Valor |
|---|---|
| Nombre canónico | Leo Mejor |
| Alias / URLs | leomejor.ai; leomejor.cl/imactiva/ |
| Origen / responsables | **Imactiva SpA** (Av. Apoquindo 6410 Of. 605, Las Condes, Santiago); contacto info@leomejor.ai |
| Trayectoria del proveedor | Empresa de desarrollo de software educativo con 16 a 20+ años, **patrocinio del MINEDUC**, presencia declarada en **+4.000 colegios** |
| Otros productos del proveedor | `Aprendiendo a leer con Bartolo` (colegios + casas), `Mi primer Bartolo`, colecciones de lenguaje y matemática |
| Público objetivo | Estudiantes y docentes de lenguaje, 1° a 8° básico |
| Plataforma | App Android nativa (no web) |
| Métricas declaradas | Velocidad lectora (PPM), precisión (correctas vs. errores), omisiones, prosodia (entonación, ritmo y expresión) |
| Biblioteca de textos | Clasificada por **niveles Lexile** para 1° a 8° básico; permite cargar lecturas propias del docente |
| Reportes | Detallados por estudiante y por curso, individuales y comparativos, historial de progreso |
| Audios | Reproducción de cada lectura grabada desde la app |
| Exportación | Excel |
| NEE | Catalogado como "adecuado a NEE" en la ficha de Por un Chile que Lee |
| Adopción declarada | **+40 colegios**, **+10.000 mediciones** realizadas (a la fecha de consulta) |
| Lanzamiento | Mediados de 2025 |
| Catálogo de referencia | Listado en **Por un Chile que Lee** (porunchilequelee.cl/producto/leo-mejor) |
| Modelo comercial | Licencia / servicio a colegios (precios no públicos) |
| Conectividad | Android; requiere dispositivo del docente o estudiante |
| Fecha de verificación | 2026-08-12 |

### 3.3 `Impulso Lector` (Agencia de Calidad, 2026)

| Atributo | Valor |
|---|---|
| Nombre | Impulso Lector |
| Origen / responsable | Agencia de Calidad de la Educación (Gobierno de Chile), dentro del plan Chile Aprende y Avanza |
| Naturaleza | Evaluación nacional, primera que mide en conjunto comprensión lectora y fluidez lectora |
| Año de implementación | 2026 |
| Público objetivo | Sistema escolar chileno (cobertura por definir por la Agencia) |
| Función | Producción de información nacional sobre nivel lector; no es una app de aula |
| Relevancia para el nicho | Genera demanda explícita de instrumentos de medición de fluidez, porque ubja la fluidez como estándar evaluativo del sistema |
| Fecha de verificación | 2026-08-12 |

### 3.4 `EPLA — Evaluación para el Progreso Lector Aptus`

| Atributo | Valor |
|---|---|
| Nombre | EPLA (Evaluación para el Progreso Lector Aptus) |
| Origen / responsables | Fundación Aptus |
| Naturaleza | Evaluación aplicada individualmente por el docente |
| Estructura | 74 tarjetas agrupadas en 4 secciones: conciencia fonológica, relación grafema-fonema, decodificación, lectura fluida |
| Función | Identificar dificultades en la ruta hacia la fluidez; diseñar estrategias de apoyo |
| Público objetivo | Primer ciclo básico |
| Tecnología | No automatizada (instrumento físico/dirigido) |
| Relevancia | Es el referente chileno de evaluación diagnóstica **estructurada** por dominios |
| Catálogo | Listado en Por un Chile que Lee y en aptus.org |
| Fecha de verificación | 2026-08-12 |

### 3.5 `Aprendiendo a leer con Bartolo` y `Mi primer Bartolo` (Imactiva)

Estos productos se reconfirman aquí porque **son del mismo proveedor que Leo Mejor** (Imactiva SpA). Es decir, Imactiva opera una línea completa de lectoescritura inicial (Bartolo, Mi primer Bartolo) más la línea de evaluación de fluidez (Leo Mejor). Eso convierte a Imactiva en un actor conglomerado dentro del ecosistema chileno, con presencia transversal en MINEDUC, Por un Chile que Lee y catálogos escolares.

---

## 4. Ecosistema internacional — fichas con evidencia nueva

### 4.1 `LUCA` (luca.ai) — ASR fonémico

| Atributo | Valor |
|---|---|
| URL | luca.ai |
| Tecnología diferenciadora | **SoundScout**: ASR a nivel **fonema** con **763.000+ mapeos grafema-fonema** |
| Funcionamiento | Cuando un niño mispronuncia, LUCA identifica **qué sonido específico** dentro de la palabra causó el error y lo mapea a un perfil continuo de dominio |
| Adaptación | Genera historias decodificables **personalizadas en tiempo real** apuntando a las habilidades específicas que el niño necesita |
| Evidencia | NSF SBIR; colaboración con Carnegie Mellon University; patente; piloto con **+17,4 WPM** de mejora |
| Público | K-adulto |
| Caso de uso óptimo | Lectores con dificultades; dislexia; precisión diagnóstica |
| Modelo | Comercial / licencia escolar |
| Fecha de verificación | 2026-08-12 |

### 4.2 `Amira-Istation` — el gigante con evidencia ESSA

| Atributo | Valor |
|---|---|
| Estado | Fusión **Amira + Istation** (junio 2024); operan como "Intelligent Growth Engine" |
| Tecnología | ASR propietario entrenado en **10 mil millones+ de palabras habladas por niños** |
| Funciones | Escucha la lectura, diagnostica, ~60 micro-intervenciones, coaching en tiempo real, generación de preguntas de comprensión |
| Evidencia | **ESSA Tier 2** con tamaño de efecto **d = 0,06-0,70** |
| Público | PreK-8 |
| Distribución en Chile | Vía **Colegium** |
| Modelo | Comercial escolar; ~USD 20-40/estudiante/año |
| Fecha de verificación | 2026-08-12 |

### 4.3 `Wadhwani AI ORF` — la prueba de escala global

| Atributo | Valor |
|---|---|
| URL | wadhwaniai.org/impact/education-solutions/oral-reading-fluency |
| Origen | Wadhwani AI (India), sector público |
| Escala declarada | **+15 millones de evaluaciones** (cifra enero 2026), **270.000+ docentes** capacitados, **96.000+ escuelas** asociadas, 7,9 M+ estudiantes |
| Cobertura | Estados de Gujarat y Rajasthan, grados 2-8 |
| Función | Escucha la lectura, marca palabras omitidas, errores y velocidad; nueva función de comprensión |
| Tecnología | ASR fine-tuneado para lenguas índicas y voces infantiles |
| Relevancia | Prueba que el modelo ORF-ASR es viable a escala país en sistema público |
| Fecha de verificación | 2026-08-12 |

### 4.4 `EGRA-AI` — consorcio África

| Atributo | Valor |
|---|---|
| URL | ai-for-education.org/lbd-egra-ai |
| Consorcio | Neurabuild, University of Cape Town, Binding Constraints Lab, Western Sydney University |
| Foco | ASR para EGRA en lenguas africanas; primero isiXhosa, extendido a **Kiswahili** |
| App de campo | **ReadUp** (adaptada para evaluación oral) |
| Extensión | Kiswahili con Zevo Tech + Laterite + Stellenbosch University |
| Lecciones publicadas | (1) una ronda de datos no basta — flexibilidad clave; (2) App refinada con detección de ruido y cambio de swipe a tap-to-speak; (3) marcado por hablantes nativos |
| Financiamiento | Grant Learning by Doing |
| Fecha de verificación | 2026-08-12 |

### 4.5 `Brasil — PUCRS / Viola et al. 2025` (Scientific Reports)

| Atributo | Valor |
|---|---|
| Paper | "Evaluation using artificial intelligence shows post pandemic differences in oral reading fluency between Brazilian public and private school students" (Sci Rep 2025) |
| Validación | Correlación **r = 0,96** entre scoring automático y humano (consistente con el estudio de Ghana, arXiv:2310.17606) |
| Hallazgo | Diferencias post-pandemia en ORF entre escuelas públicas y privadas en Brasil |
| Beneficio reportado | Evaluación de toda una clase en **menos de 20 minutos** sin pérdida de precisión |
| Relevancia | Validación en **portugués** — la lengua más cercana tipológicamente al español de Chile |

### 4.6 `NWEA MAP Reading Fluency`

| Atributo | Valor |
|---|---|
| Origen | NWEA (EE. UU.) |
| Naturaleza | Evaluación online adaptativa, universal screener PreK-5 (extensible a 6-8) |
| Cobertura | Fluidez oral, comprensión, habilidades fundacionales |
| Frecuencia típica | 3 veces al año para benchmarking |
| Modelo | Comercial / institucional |
| Fecha de verificación | 2026-08-12 |

### 4.7 `Amplify mCLASS TextReadingOnline (TRO)`

| Atributo | Valor |
|---|---|
| Origen | Amplify (EE. UU.) |
| Naturaleza | Solución de alfabetización con voz para grados 1-6 |
| Funciones | Evaluación remota de ORF, precisión y comprensión |
| Diferenciador | Reconocimiento de voz para medición de fluidez; parte del ecosistema mCLASS / DIBELS 8th Edition |
| Fecha de verificación | 2026-08-12 |

### 4.8 `ReadingFluency.ai` y `ReadingFluency.app`

| Atributo | Valor |
|---|---|
| URLs | readingfluency.ai; readingfluency.app |
| Naturaleza | Herramienta de scoring WCPM con IA |
| Función | Una lectura en voz alta de 60 segundos entrega un puntaje exacto contra benchmarks nacionales |
| Público | K-5 (escuela, curso, hogar) |
| Modelo | Freemium / libre con registro |
| Fecha de verificación | 2026-08-12 |

### 4.9 `ReadFlare`

| Atributo | Valor |
|---|---|
| URL | readflare.com |
| Naturaleza | Explicación de tests de fluidez (ORF, WCPM, Acadience, DIBELS), derechos de las familias |
| Función | Más orientada a comprensión de resultados que a evaluación directa |
| Fecha de verificación | 2026-08-12 |

### 4.10 `ClearFluency` (anteriormente Reading Assistant)

| Atributo | Valor |
|---|---|
| Origen | Discovery Education / Triumph Learning (EE. UU.) |
| Naturaleza | Práctica guiada de lectura oral con feedback |
| Listado | Aparece en comparativos de software de fluidez 2026 |
| Fecha de verificación | 2026-08-12 |

### 4.11 `SoapBox Labs` → `Curriculum Associates` → `i-Ready`

SoapBox Labs, el motor ASR especializado en voces infantiles (Irlanda), fue **adquirido por Curriculum Associates**. Su tecnología se integró en **i-Ready**, el programa de lectura basal más difundido en EE. UU. Esto consolida el ASR infantil dentro del canal comercial más grande de Norteamérica. Implicación: la barrera de entrada de un motor ASR infantil propietario subió; el espacio para diferenciarse se mueve al **cómo se usa** la transcripción, no a **tener** transcripción.

### 4.12 `Plabook` (Data Monsters / NVIDIA Riva)

| Atributo | Valor |
|---|---|
| Origen | Data Monsters, sobre NVIDIA Riva |
| Funciones | Automatiza evaluación de fluidez oral y detección de dislexia con reportes |
| Público | Escuelas, EdTech, docentes |
| Fecha de verificación | 2026-08-12 |

### 4.13 `Flapp` y `DECILE` (Argentina, BID)

Confirmados y ampliados por el documento del BID *Alfabetización + Inteligencia Artificial (IA). Desafíos y oportunidades en Argentina*:

- **Flapp** — Plataforma de Evaluación de Fluidez Lectora para preescolar, 1° y 2° grado; diagnóstico rápido vía IA; permite al docente identificar dificultades específicas y personalizar intervención. Resultados construidos sobre censos de fluidez lectora de Mendoza; recomendación de entrenar con datos locales.
- **DECILE** — Herramienta de IA para desarrollo del lenguaje infantil; identifica trayectorias variables en niños con dificultades auditivas; análisis grupal e individual; genera métricas para intervención personalizada.

### 4.14 `Microsoft Reading Progress`, `Reading Coach`, `Lector Inmersivo`

Sin cambios estructurales. Siguen siendo la capa gratuita dentro del ecosistema Microsoft (Teams / Edge). Confirmados por la nota oficial de Microsoft Latinoamérica de uso en aula (caso República Dominicana).

### 4.15 `Google Read Along`

Mantiene su rol de apoyo escolar con asistente virtual (Diya). La línea de Google Workspace (Education Plus, Teaching and Learning Upgrade) sigue siendo relevante para la operación de aula.

---

## 5. Estado del arte académico

### 5.1 ORF con ASR en niños (línea principal)

| Referencia | Aporte | Dato clave |
|---|---|---|
| **van der Velde et al. 2025** (Int J Artif Intell Educ, Springer) "Speech Enabled Reading Fluency Assessment: a Validation Study" | Validación ARGUMENT-BASED de un instrumento ORF basado en ASR para lengua semi-transparente (holandés) | 176 h de audio, 653 niños de 2°-3°, 569 y 622 pruebas palabra/pasaje; **demostrada validez** para decodificación, velocidad y automaticidad |
| **arXiv:2310.17606** Henkel et al. 2023 "Using State-of-the-Art Speech Models to Evaluate Oral Reading Fluency in Ghana" | Whitney V2 out-of-the-box para ORF en Global South | WER **13,5** en lectura infantil ghanés; correlación **r=0,96** automatizado vs. experto humano |
| **arXiv:2405.19426** Vaidya, Sahoo, Rao 2024 "Deep Learning for Assessment of Oral Reading Fluency" | wav2vec2.0 end-to-end sobre audio infantil etiquetado por expertos | Probing de embeddings para léxico y acústico-prosódico |
| **arXiv:2306.03444** Molenaar et al. 2023 "Automatic Assessment of Oral Reading Accuracy for Reading Diagnostics" | Seis sistemas ASR (Kaldi, Whisper) para lectura holandesa | Mejor sistema: MCC = 0,63; incluir errores de lectura en el LM mejora desempeño |
| **arXiv:2606.31508** Diarra et al. 2026 "Building an ASR Solution for Training and Assessing Children's Reading" (Bambara) | Sistema end-to-end abierto: campo + benchmark + modelo + app + validación aula | **Soloni** Fast-Conformer (TDT + CTC) reduce WER de 0,42 a 0,22; niños < 10 son la fuente principal del error residual |
| **arXiv:2507.13205** Louw et al. 2025 "Automatically assessing oral narratives of Afrikaans and isiXhosa children" | ASR + scoring ML/LLM de narrativas orales preescolares | LLM iguala a experto humano en **identificar niños que requieren intervención** |
| **arXiv:2504.20678** Getman et al. 2025 "NOCASA — Non-native Children's Automatic Speech Assessment Challenge" (IEEE MLSP 2025) | Challenge público de evaluación de pronunciación infantil L2 | Dataset TeflonNorL2 (10.334 grabaciones, 44 hablantes, 205 palabras noruegas); baseline wav2vec2.0 UAR 36,37% |

### 5.2 Detección de mispronunciación y diagnóstico (MDD) — la frontera técnica

| Referencia | Aporte | Dato clave |
|---|---|---|
| **arXiv:2604.22133** Geng et al. 2026 "Beyond Acoustic Sparsity and Linguistic Bias: A Prompt-Free Paradigm for MDD" | Marco **prompt-free** que desacopla acústica de priors canónicos; modelo **CROTTC** con alineamiento monotónico frame-level | F1 71,77% en L2-ARCTIC; 71,70% en leaderboard Iqra'Eval2 |
| **arXiv:2606.22022** Chen, Shahin, Ahmed 2026 "Phonological-Level Wav2Vec2 for Mandarin MDD" | MDD por **rasgos fonológicos** (segmentales + tonales) en wav2vec2 CTC unificado | Reduce FAR 10,1% y DER 23,6% vs. baseline por fonema |
| **arXiv:2511.20107** Tu et al. 2025 "MDD Without Model Training: A Retrieval-Based Approach" | MDD **sin entrenamiento** vía retrieval + ASR pre-entrenado | F1 69,60% en L2-ARCTIC |
| **arXiv:2506.12067** Parikh et al. 2025 "Evaluating Logit-Based GOP Scores for MDD" | GOP basado en logits (no probabilidades) para pronunciation assessment | Logit máximo es el mejor alineado con percepción humana |
| **arXiv:2311.07037** Shahin, Epps, Ahmed 2023 "Phonological Level wav2vec2-based MDD" | MDD por atributos de habla (rasgos articulatorios) con multi-label CTC | Baja FAR, FRR y DER sobre todos los atributos vs. enfoque por fonema |
| **arXiv:2406.04595** Wang et al. 2024 "Pitch-Aware RNN-T for Mandarin Chinese MDD" | RNN-T stateless con HuBERT + Pitch Fusion Block | +3% Phone Error Rate, +7% FAR vs. SOTA en escenarios no nativos |

### 5.3 Casos especiales y complementarios

| Referencia | Aporte |
|---|---|
| **arXiv:2305.16085** Benway et al. 2023 — Inversión acústico-a-articulatoria para detección de mispronunciación de /r/ en Speech Sound Disorders infantiles | Iguala o supera SOTA al predecir rhoticidad clínica |
| **arXiv:2402.15539** Lee et al. 2024 — Corpus de habla para niños coreanos con TEA | Modelo de corpus clínico infantil replicable |
| **arXiv:2607.22377** Fadurudeen 2026 — **Kutti AI**: compañero de aprendizaje voz-first offline para niños con discapacidad visual | Detección de dificultad en tiempo real; fuzzy matching cross-lingüe; ASR on-device para baja conectividad |
| **arXiv:2509.22287** Sundstedt et al. 2025 — TalBot: robot conversacional Furhat con LLM para niños preescolares con vulnerabilidades del lenguaje | Generación LLM de objetivos morfológicos durante el juego "Alias"; modelo de rol docente/SLT |
| **Frontiers in Education 2026** "Evaluation of the consistency of a speech verification system with human raters in early literacy screening assessments" | Validación de consistencia entre sistema de verificación de voz y raters humanos en screening lector temprano |
| **Turner et al. 2025** (Res Methods Appl Linguist) "Evaluating the scoring system of an AI-integrated app to assess foreign language phonological decoding" | Evaluación del scoring de app con IA para decodificación fonológica en L2 |

---

## 6. Tabla maestra de recursos nuevos

> Consolidado en formato reutilizable.

| Recurso / alias | Rol en esta síntesis | Categoría | Origen / responsable | Público objetivo | Síntesis funcional | IA / diferenciador | Acceso / conectividad | Origen del dato |
|---|---|---|---|---|---|---|---|---|
| `PerroLeo` / `Perro Leo` | **Comparación directa Chile** | Evaluación automatizada de fluidez | PerroLeo en alianza con Educarchile (Fundación Chile + MINEDUC) | 1°-8° básico, docentes | Captura multi-dispositivo de lectura oral; entrega velocidad, calidad y nivel de logro al instante; rúbrica validada | Sí; ASR + rúbrica pedagógica | Licencias (convocatoria gratuita 2° semestre 2026 para 2° y 4° básico); web; Ley 21.719 | ET |
| `Leo Mejor` | **Comparación directa Chile** | Evaluación automatizada de fluidez | Imactiva SpA (Chile) | 1°-8° básico, docentes de lenguaje | App Android; velocidad, precisión, omisiones y prosodia; textos Lexile; reportes y exportación Excel; reproduce audios | Sí; reconocimiento de voz | Comercial / servicio a colegios; +40 colegios, +10K mediciones; adecuado a NEE | ET |
| `Impulso Lector` | Prioritario Chile (marco) | Evaluación nacional | Agencia de Calidad de la Educación (Chile) | Sistema escolar chileno | Primera evaluación nacional 2026 que mide conjuntamente comprensión y fluidez lectora | No (evaluación系统性) | Institucional; nacional | ET |
| `EPLA` | Complementario Chile | Diagnóstico lector por dominios | Fundación Aptus (Chile) | Primer ciclo básico, docentes | 74 tarjetas en 4 secciones: conciencia fonológica, grafema-fonema, decodificación, fluidez | No (instrumento dirigido) | Gratuito; físico/dirigido | ET |
| `Aprendiendo a leer con Bartolo` / `Mi primer Bartolo` (confirmado) | Complementario Chile | Software de apoyo a lectoescritura | Imactiva SpA (Chile) con patrocinio MINEDUC | 1°-3° básico, familias | Software de lectoescritura inicial en modelo equilibrado | No; lúdico | Comercial; Android; mixto | ET |
| `LUCA` / `luca.ai` | Comparado | IA y lectura oral con MDD fonémico | LUCA (EE. UU.) + CMU + NSF | K-adulto, lectores con dificultades, dislexia | ASR **SoundScout** a nivel fonema; 763K+ mapeos grafema-fonema; genera historias decodificables personalizadas | Sí; ASR fonémico + generación adaptativa | Comercial; online; NSF SBIR; +17,4 WPM piloto | ET |
| `Amira-Istation` | Comparado | IA y tutoría de lectura oral | Amira + Istation (EE. UU.); distribución Chile vía Colegium | PreK-8 | Tutor IA con ~60 micro-intervenciones; ASR propio entrenado con 10B+ palabras infantiles | Sí; ASR propietario; ESSA Tier 2 (d=0,06-0,70) | Comercial; ~USD 20-40/estudiante/año | ET |
| `Wadhwani AI ORF` | Comparado / benchmark escala | IA y ORF en sistema público | Wadhwani AI (India) | Grados 2-8, sistema público | Escucha lectura; marca palabras omitidas, errores y velocidad; nueva función de comprensión | Sí; ASR fine-tuneado para lenguas índicas | Institucional; **+15M evaluaciones**, 270K docentes, 96K escuelas (ene 2026) | ET |
| `EGRA-AI` | Benchmark regional | ASR para EGRA en lenguas africanas | Neurabuild + U. Cape Town + Binding Constraints Lab + Western Sydney U | Niños de grados iniciales en lenguas africanas | ASR + app ReadUp;adapta el modelo EGRA uno-a-uno a uno-a-muchos | Sí; ASR + detección de ruido | Consorcio; isiXhosa→Kiswahili | ET |
| `Brasil PUCRS ORF` / `Viola et al. 2025` | Referencia indirecta | Validación científica ORF en portugués | PUCRS (Brasil), Sci Rep | Sistema escolar brasileño | Validación de ORF con IA post-pandemia; clase evaluada en <20 min | Sí; ASR; **r=0,96** vs. humano | Investigación; brasileño/portugués | ET |
| `NWEA MAP Reading Fluency` | Comparado | Evaluación adaptativa ORF | NWEA (EE. UU.) | PreK-5 (ext. 6-8) | Universal screener; ORF + comprensión + habilidades fundacionales; 3 veces al año | Sí; adaptativo | Comercial; institucional | ET |
| `Amplify mCLASS TextReadingOnline (TRO)` | Comparado | Evaluación ORF con voz | Amplify (EE. UU.) | Grados 1-6 | Evaluación remota de ORF, precisión y comprensión; parte del ecosistema mCLASS/DIBELS 8 | Sí; reconocimiento de voz | Comercial; institucional | ET |
| `ReadingFluency.ai` / `ReadingFluency.app` | Comparado | Scoring WCPM con IA | Privado (EE. UU.) | K-5 | Una lectura de 60 s entrega score WCPM contra benchmarks nacionales | Sí; IA de scoring | Freemium; online | ET |
| `ReadFlare` | Referencia indirecta | Explicación de tests ORF | Privado (EE. UU.) | Familias y docentes | Recursos para entender ORF, WCPM, Acadience, DIBELS | No; divulgativo | Gratuito; online | ET |
| `ClearFluency` | Comparado | Práctica guiada de lectura oral | Discovery Education / Triumph (EE. UU.) | Escolar | Práctica de lectura oral con feedback | Sí/asistiva | Comercial; online | ET |
| `SoapBox Labs → Curriculum Associates → i-Ready` | Comparado | Infraestructura ASR infantil | SoapBox Labs (Irlanda) → Curriculum Associates (EE. UU.) | Desarrolladores EdTech, escuelas | Motor ASR infantil integrado en i-Ready tras adquisición | Sí; ASR infantil | Comercial; institucional | ET |
| `Plabook` | Comparado | IA y ORF + detección dislexia | Data Monsters sobre NVIDIA Riva (EE. UU.) | Escuelas, EdTech, docentes | Automatiza ORF y detección de dislexia con reportes | Sí; voz + detección | Comercial; online | ET |
| `Flapp` | Benchmark regional | Evaluación de fluidez (preescolar-2°) | Argentina (BID) | Preescolar, 1°, 2° | Diagnóstico rápido con IA; intervención personalizada | Sí; IA; validada con censos de Mendoza | Piloto; institucional | ET |
| `DECILE` | Benchmark regional | Desarrollo del lenguaje infantil con IA | Argentina (BID) | Niños con desarrollo lingüístico típico y dificultades | Métricas grupales e individuales; trayectorias variables en niños con dificultades auditivas | Sí; IA | Piloto; institucional | ET |

---

## 7. Fuentes

### 7.1 Sitios oficiales y notas de prensa (verificados 2026-08-12)

- PerroLeo: https://www.perroleo.cl/
- Educarchile — Convocatoria PerroLeo: https://www.educarchile.cl/articulos/conoce-perro-leo-y-participa-por-licencias-que-te-permitiran-evaluar-la-fluidez-y-calidad
- Educarchile — Formulario convocatoria: https://www.educarchile.cl/articulos/participa-por-licencias-gratuitas-que-te-permitiran-evaluar-la-fluidez-y-calidad-lectora
- Leo Mejor: https://www.leomejor.ai/
- Leo Mejor (sitio alternativo): https://www.leomejor.cl/
- Imactiva SpA: https://www.imactiva.cl/
- Imactiva Colegios: https://www.imactiva.cl/colegios/
- Por un Chile que Lee — Ficha Leo Mejor: https://porunchilequelee.cl/producto/leo-mejor
- Por un Chile que Lee — Ficha EPLA: https://porunchilequelee.cl/producto/evaluacion-para-el-progreso-lector-epla
- Aptus — EPLA: https://www.aptus.org/producto/evaluacion-para-el-progreso-lector-aptus-epla
- Agencia de Calidad — Impulso Lector: comunicado y video en redes oficiales de la Agencia de Calidad de la Educación (2026)
- LUCA: https://luca.ai/blog/best-ai-reading-tutors
- Amira Learning: https://explore.amiralearning.com/
- Wadhwani AI ORF: https://www.wadhwaniai.org/impact/education-solutions/oral-reading-fluency
- EGRA-AI Consorcio: https://ai-for-education.org/lbd-egra-ai
- EGRA-AI Grand Challenges: https://gcgh.grandchallenges.org/grant/automating-early-grade-reading-assessments-egra-african-languages-using-voice-recognition-ai
- ADEA — AI in foundational learning: https://www.adeanet.org/en/blogs/how-artificial-intelligence-shaping-foundational-learning-assessments-africa
- NWEA MAP Reading Fluency: https://www.nwea.org/map-reading-fluency/
- Amplify mCLASS TRO: https://amplify.com/mclass-tro/
- ReadingFluency.ai: https://readingfluency.ai/
- ReadingFluency.app: https://readingfluency.app/
- ReadFlare: https://readflare.com/articles/assessment-and-testing/reading-fluency-test
- Tools Competition — ORF Hindi: https://tools-competition.org/winner/oral-reading-fluency
- The Learning Agency — ASR para lectura: https://the-learning-agency.com/the-cutting-ed/article/teaching-kids-to-read-with-speech-recognition-technology
- International Education News — Micro-innovations in assessment (2026-07-15): https://internationalednews.com/2026/07/15/micro-innovations-in-assessment-for-specific-subjects-levels-and-contexts-ai-new-technologies-and-the-future-of-assessment-part-4
- BID — Alfabetización + IA Argentina: https://publications.iadb.org/publications/spanish/document/Alfabetizacion-inteligencia-artificial-IA.-Desafios-y-oportunidades-en-Argentina.pdf
- Microsoft LATAM — Reading Progress + Lector Inmersivo: https://news.microsoft.com/es-xl/la-inteligencia-artificial-en-herramientas-gratuitas-como-reading-progress-y-lector-inmersivo-permite-mejorar-el-desempeno-de-lectura-en-estudiantes

### 7.2 Literatura científica (validación externa)

- van der Velde, M., Harmsen, W., Veldkamp, B. P., Feskens, R., Keuning, J., Swart, N. (2025). Speech enabled reading fluency assessment: a validation study. International Journal of Artificial Intelligence in Education, 35, 2569-2595. https://doi.org/10.1007/s40593-025-00480-y — PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12686063
- Viola, T. W. et al. (2025). Evaluation using artificial intelligence shows post pandemic differences in oral reading fluency between Brazilian public and private school students. Scientific Reports, 15, 30131. https://doi.org/10.1038/s41598-025-15644-y — PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12358507
- Frontiers in Education (2026). Evaluation of the consistency of a speech verification system with human raters in early literacy screening assessments. https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2026.1671946/full
- Turner, J. et al. (2025). Evaluating the scoring system of an AI-integrated app to assess foreign language phonological decoding. Research Methods in Applied Linguistics, 4, 100257. https://doi.org/10.1016/j.rmal.2025.100257

### 7.3 Literatura arXiv

- arXiv:2310.17606 — Henkel et al. 2023, ORF con Whisper V2 en Ghana
- arXiv:2405.19426 — Vaidya, Sahoo, Rao 2024, Deep Learning ORF (wav2vec2)
- arXiv:2306.03444 — Molenaar et al. 2023, ASR para ORF holandesa
- arXiv:2606.31508 — Diarra et al. 2026, ASR Bambara para lectura infantil
- arXiv:2507.13205 — Louw et al. 2025, narrativas orales Afrikaans/isiXhosa
- arXiv:2504.20678 — Getman et al. 2025, NOCASA challenge noruego
- arXiv:2604.22133 — Geng et al. 2026, MDD prompt-free CROTTC
- arXiv:2606.22022 — Chen, Shahin, Ahmed 2026, MDD fonológico Mandarin
- arXiv:2511.20107 — Tu et al. 2025, MDD retrieval training-free
- arXiv:2506.12067 — Parikh et al. 2025, GOP logit-based MDD
- arXiv:2311.07037 — Shahin, Epps, Ahmed 2023, MDD fonológico wav2vec2
- arXiv:2406.04595 — Wang et al. 2024, Pitch-Aware RNN-T Mandarin MDD
- arXiv:2305.16085 — Benway et al. 2023, inversión acústico-a-articulatoria para /r/ SSD
- arXiv:2402.15539 — Lee et al. 2024, corpus habla infantil TEA coreano
- arXiv:2607.22377 — Fadurudeen 2026, Kutti AI voz-first
- arXiv:2509.22287 — Sundstedt et al. 2025, TalBot robot LLM

### 7.4 Nota metodológica

- Las fichas se construyeron a partir de sitios oficiales, prensa institucional y literatura indexada. Las afirmaciones declarativas ("+40 colegios", "+15M evaluaciones", "r=0,96") se atribuyen a la fuente original y no fueron auditadas independientemente por este reporte.
- Las lecturas estratégicas son interpretativas y no constituyen recomendación de adopción ni de compra.
- Las fechas de verificación correspondientes a "2026-08-12" indican el día de la consulta, no la última actualización del sitio fuente.
