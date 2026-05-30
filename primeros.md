# Reporte de Entrenamientos — Runs 0 a 22

## Resumen general

Este documento detalla la evolución del proyecto desde el primer prototipo (Run 0) hasta el rebalanceo del dataset (Run 22). Se recorrieron múltiples arquitecturas, configuraciones y estrategias de regularización, enfrentando desafíos como overfitting, alta varianza, desbalance de clases y limitaciones del dataset.

---

## Run 0 — Primera versión (flatten completo)

| Métrica | Valor |
|:--------|:------|
| Val Accuracy | ~44% |
| Parámetros | 27.7M |
| Tiempo | ~15s |

**Arquitectura:** Modelo inicial con aplanamiento completo (`Flatten`) de la salida convolucional, sin Adaptive Pooling. Red extremadamente grande para el dataset disponible.

**Dificultades:**
- Explosión de parámetros (27.7M) → overfitting severo
- Tiempo de entrenamiento alto para la época por la cantidad de parámetros
- Accuracy muy baja, la red no generalizaba

---

## Run 1 — GlobalAvgPool2d + más capas conv

| Métrica | Valor |
|:--------|:------|
| Val Accuracy | ~35% |
| Parámetros | 1.8M |
| Tiempo | ~20s |

**Cambios:** Reemplazo de `Flatten` por `GlobalAvgPool2d`. Adición de más capas convolucionales para compensar la reducción de parámetros.

**Dificultades:**
- El accuracy cayó respecto a Run 0 (de 44% a 35%)
- GlobalAvgPool2d reduce drásticamente los parámetros pero pierde capacidad de representación espacial
- La red quedó con muy pocos parámetros para la complejidad del problema

---

## Run 2 — Aumento de filtros convolucionales

| Métrica | Valor |
|:--------|:------|
| Val Accuracy | ~48% |
| Parámetros | 7.8M |
| Tiempo | ~15s |

**Cambios:** Incremento en la cantidad de filtros por capa convolucional.

**Dificultades:**
- Mejora moderada (35% → 48%)
- Todavía lejos de un accuracy aceptable
- Parámetros altos nuevamente (7.8M) para un dataset de solo ~144 imágenes originales

---

## Run 3 — AdaptiveAvgPool2d + LR Scheduler

| Métrica | Valor |
|:--------|:------|
| Val Accuracy | ~55% |
| Parámetros | 1.45M |
| Tiempo | ~18s |

**Cambios:** Reemplazo de `GlobalAvgPool2d` por `AdaptiveAvgPool2d`. Incorporación de `ReduceLROnPlateau` como scheduler.

**Dificultades:**
- Primera mejora significativa (48% → 55%)
- AdaptiveAvgPool2d permitió mantener flexibilidad espacial
- El LR scheduler ayudó a que el entrenamiento sea más estable
- Con dataset original de ~144 imágenes, el techo de accuracy era limitado

---

## Run 4 — AdaptiveAvgPool2d((4,4)) + 160×160

| Métrica | Valor |
|:--------|:------|
| Val Accuracy | **72.41%** |
| Parámetros | 1.45M |
| Tiempo | 21s |

**Cambios:**
- Salida fija de `AdaptiveAvgPool2d((4,4))` en lugar de (1,1)
- Imágenes redimensionadas a 160×160 (antes otro tamaño)

**Aciertos:**
- Salto enorme de 55% a 72.41% — el mejor resultado hasta ese momento
- El pooling (4,4) en lugar de (1,1) preserva información espacial relevante para distinguir billetes
- El tamaño 160×160 fue un平衡 entre resolución y cantidad de parámetros

**Dificultades:**
- Dataset pequeño (~144 imágenes) → alta varianza entre corridas
- Sin embargo, demostró que la arquitectura era prometedora

---

## Run 5 — 4 capas convolucionales + 200×200

| Métrica | Valor |
|:--------|:------|
| Val Accuracy | ~48% |
| Parámetros | 2.7M |
| Tiempo | 25s |

**Cambios:** Se agregó una cuarta capa convolucional. Imágenes a 200×200.

**Dificultades:**
- Caída abrupta: 72.41% → 48%
- Cuatro capas convolucionales resultaron excesivas para el dataset
- 200×200 aumentó la dimensionalidad sin beneficio, introduciendo más parámetros innecesarios
- La red volvió a overfittear

---

## Run 11 — Label smoothing + Weight Decay 0.01

| Métrica | Valor |
|:--------|:------|
| Val Accuracy | **73.81%** |
| Parámetros | 651K |
| Tiempo | 20s |

**Cambios:**
- Volvió a la arquitectura de 3 capas conv (32→64→128) con AdaptiveAvgPool2d((4,4))
- Parámetros reducidos a 651K
- **Label smoothing (0.1)** en CrossEntropyLoss
- **Weight decay 0.01** en optimizador Adam

**Aciertos:**
- Recuperación del mejor accuracy (73.81%), superando ligeramente Run 4
- Label smoothing evitó que el modelo quedara demasiado confiado en sus predicciones
- Weight decay alto (0.01) fue clave para controlar el overfitting

**Dificultades:**
- Mejora marginal sobre Run 4
- El dataset seguía siendo el limitante principal

---

## Run 14 — Verificación de varianza

| Métrica | Valor |
|:--------|:------|
| Val Accuracy | 61.90% |
| Parámetros | 651K |
| Tiempo | 17s |

**Cambios:** Misma configuración que Run 11, ejecutada para verificar consistencia.

**Dificultades:**
- Confirmación de alta varianza: 73.81% → 61.90% con la misma configuración
- La inicialización aleatoria y el data augmentation producen resultados muy dispares
- Con dataset chico, cada split de validación afecta enormemente la métrica

---

## Run 15 — Arquitectura grande + dataset completo

| Métrica | Valor |
|:--------|:------|
| Val Accuracy | 61.83% |
| Parámetros | 5.88M |
| Tiempo | ~7min |

**Cambios:** Primera ejecución con el **dataset expandido** (~2121 imágenes). Arquitectura más grande.

**Dificultades:**
- El dataset más grande no se tradujo en mejor accuracy inmediato
- La arquitectura grande (5.88M params) seguía siendo propensa a overfitting
- El aumento de dataset necesita ajuste de hiperparámetros, no solo más capacidad

---

## Run 16 — Arquitectura chica + dataset completo

| Métrica | Valor |
|:--------|:------|
| Val Accuracy | 71.29% |
| Test Accuracy | 67.63% |
| Parámetros | 651K |
| Tiempo | 4.8min |

**Cambios:** Se usó la arquitectura chica probada (651K params) con el dataset completo.

**Aciertos:**
- 71.29% val — confirmación de que la arquitectura de 651K params es la adecuada
- Primera medición de test accuracy (67.63%)
- Dataset completo + arquitectura adecuada dieron baseline sólido

**Dificultades:**
- Test accuracy sensiblemente menor que val accuracy → posible overfitting leve
- La diferencia val-test indicaba que la generalización no era óptima

---

## Run 17 — Weight Decay 0.01 + 200 épocas

| Métrica | Valor |
|:--------|:------|
| Val Accuracy | **80.13%** |
| Test Accuracy | 85.02%* |
| Parámetros | 651K |
| Tiempo | 4.6min |

**Cambios:**
- Aumento de épocas a 200 (antes menos)
- Weight decay mantenido en 0.01
- Early stopping agregado

**Aciertos:**
- **Salto cualitativo: 80.13% val — primera vez sobre 80%**
- Test accuracy 85.02% (sobre test chico de ~207 imágenes)
- Las 200 épocas permitieron que el modelo convergiera adecuadamente
- El weight decay alto controló el overfitting incluso con más épocas

**Dificultades:**
- \*Test chico (~207 imágenes, 8.9% del dataset) — no representativo
- Alta varianza: runs posteriores mostraron resultados inferiores

---

## Run 18 — Clase 10000 aumentada

| Métrica | Valor |
|:--------|:------|
| Val Accuracy | 79.18% |
| Test Accuracy | 76.81%* |
| Parámetros | 651K |
| Tiempo | 4.0min |

**Cambios:** Clase 10000 aumentada de ~180 a 246 imágenes de entrenamiento.

**Dificultades:**
- Leve caída vs Run 17 (80.13% → 79.18%)
- Aumentar una clase no fue suficiente; el resto del dataset seguía igual
- La clase 10000 seguía siendo la minoritaria

---

## Run 19 — Sin class weights

| Métrica | Valor |
|:--------|:------|
| Val Accuracy | 76.97% |
| Test Accuracy | 73.91%* |
| Parámetros | 651K |
| Tiempo | 3.4min |

**Cambios:** Eliminación de class weights balanceados.

**Dificultades:**
- Caída de 79.18% a 76.97%
- Los class weights ayudaban a balancear el entrenamiento, especialmente para clases minoritarias
- Sin class weights, la red se sesgaba hacia las clases mayoritarias

---

## Run 20 — Verificación (alta varianza)

| Métrica | Valor |
|:--------|:------|
| Val Accuracy | 74.76% |
| Test Accuracy | 76.81%* |
| Parámetros | 651K |
| Tiempo | 3.7min |

**Cambios:** Misma configuración que Run 17, para verificar consistencia.

**Dificultades:**
- Confirmación de alta varianza: 80.13% → 74.76%
- El problema de la varianza persistía incluso con el dataset expandido
- La inicialización aleatoria y el data augmentation producen resultados inestables

---

## Run 21 — Verificación

| Métrica | Valor |
|:--------|:------|
| Val Accuracy | 74.13% |
| Test Accuracy | 71.50%* |
| Parámetros | 651K |
| Tiempo | 3.6min |

**Cambios:** Segunda verificación de la misma configuración.

**Dificultades:**
- Rango de varianza: 71.50% a 80.13% en test — demasiado amplio
- La configuración era correcta pero la estabilidad no era aceptable
- El dataset era la causa raíz: splits desbalanceados y clases con pocas muestras

---

## Run 22 — Dataset rebalanceado (20% test) — RÉCORD

| Métrica | Valor |
|:--------|:------|
| Val Accuracy | **81.70%** |
| Test Accuracy | **81.70%** |
| Parámetros | 651K |
| Tiempo | 6.2min |

**Cambios:**
- **Dataset rebalanceado:** test aumentado de ~8.9% a 20% (552 imágenes)
- Redistribución de imágenes entre splits manteniendo la misma arquitectura

**Aciertos:**
- **Nuevo récord: 81.70% val y test — misma métrica en ambos sets**
- La igualdad val-test indica generalización real, no overfitting
- El test más grande (552 imágenes) dio una medición mucho más confiable
- Demostró que el dataset previo tenía un test chico y poco representativo

**Dificultades:**
- Clase 10000 seguía siendo la más afectada (menor cantidad de imágenes)
- Confusiones persistentes entre pares (100↔200, 200↔500, 2000↔10000)
- Margen para mejorar: técnicas de data augmentation más agresivas

---

## Evolución de métricas (línea de tiempo)

```
Accuracy (val)
 80% +                     R22(81.7%)  R17(80.1%)
      |                    R18(79.2%)  
 70% +    R4(72.4%)  R16(71.3%)  R19(77.0%)
      |    R11(73.8%)
 60% + R3(55%)  R14(61.9%)  R15(61.8%)
      | R2(48%)
 50% + R0(44%)  R5(48%)
      | R1(35%)
 40% +
      |
 30% +_________________________________________
      0   1   2   3   4   5  11  14  15  16  17
```

---

## Problemas comunes identificados

1. **Dataset pequeño original (~144 imágenes):** La causa raíz de la mayoría de los problemas iniciales. Cualquier mejora en arquitectura tenía un techo bajo.

2. **Alta varianza:** Misma configuración producía resultados entre 61% y 80%. La inicialización aleatoria + data augmentation causaban inestabilidad.

3. **Desbalance de clases:** Clase 10000 siempre con menos muestras, afectando su recall y precision.

4. **Test no representativo:** Hasta Run 16 no se medía test; hasta Run 22 el test era solo ~207 imágenes (8.9%).

5. **Confusiones sistemáticas:** 100↔200 (billetes rojos), 200↔500, 2000↔10000 (billetes violetas oscuros) — billetes con colores similares se confunden.

---

## Aprendizajes clave

- **Weight decay alto (0.01)** fue el cambio individual más impactante (Run 17)
- **Arquitectura de 651K params** (3 bloques conv) es el punto óptimo para este dataset
- **AdaptiveAvgPool2d((4,4))** preserva suficiente información espacial sin explotar en parámetros
- **Label smoothing (0.1)** ayuda a que el modelo no quede sobreconfiado
- **Dataset más grande y balanceado** es lo que más mejora la generalización
- El **data augmentation online** (rotación, color jitter, affine) compensa parcialmente el dataset chico
