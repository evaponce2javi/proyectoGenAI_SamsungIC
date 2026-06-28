# proyectoGenAI_SamsungIC

# Mejoras al proyecto — Mobile Recommendation
Se construye un **recomendador de smartphones basado en personas** usando un Weighted Sum Model sobre `sic_mobile_spec.xlsx` . 


## Diagnóstico del original
El notebook *TEACHER* corre sin caerse, pero tiene fallos de correctitud, robustez y
explicabilidad. La rúbrica del propio curso premia *transparencia/explicabilidad del
peso* y *reproducibilidad*, justo donde estaba más débil.

## Cambios aplicados

1. **Columna camera sucia.** Mezcla escalas (valores 0, 2, 3 … 108, 180, 200) y usa 0
   —un valor imposible en MP— como "peor cámara", distorsionando a la persona *Creator*.
   → parse_camera trata 0 como dato faltante (NaN) y se reporta cuántos hay (1 fila).

2. **min_max_scale sin protección.** (x-min)/(max-min) produce NaN (0/0) en columnas
   constantes. → Se añadió guarda que devuelve 0.0 en ese caso.

3. **Bug de nombre en la UI.** El recomendador pedía storage_norm (no existe) y creaba una
   columna fantasma 0.0 nunca usada. → Corregido a storage_clean_norm.

4. **Sin validación de pesos.** Nada comprobaba que los pesos de cada persona sumaran 1.0 ni
   que las columnas existieran; un error tipográfico mis-escalaba en silencio.
   → validate_weights() revisa la suma (≈1.0), **renormaliza** si hace falta y avisa de
   columnas ausentes.

5. **Score opaco (explicabilidad).** El recomendador daba un único Score sin justificación.
   → explain_score() desglosa la **contribución ponderada** de cada característica al puntaje
   de un teléfono (p. ej. para el #1 de *Creator*: cámara 40 + storage 25 + rating 10.5 +
   amoled 10 + ram 10 = 95.5).

6. **Solo funcionaba con kernel vivo.** La UI de ipywidgets no produce nada al exportar a
   PDF/HTML (lo que un instructor usaría para corregir).
   → Se añadió una **salida estática Top-5 por persona** que sobrevive a la exportación,
   manteniendo además la UI interactiva.

7. **EDA poco legible.** Imprimía una Series de correlaciones cruda.
   → Mapa de calor de correlaciones + correlaciones ordenadas frente a price_clean.

8. **Sin verificación de validez.** Nueva sección de **validez de cara**: comprueba que el
   teléfono #1 de cada persona puntúe al menos tan alto bajo su propia persona como bajo las
   demás. Reveló un hallazgo real: *Student* y *Business_Professional* solapan con otras
   personas (su top gusta más a un perfil distinto), señal de que esos pesos podrían afinarse.
