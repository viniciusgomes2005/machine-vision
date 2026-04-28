Snapshot da iteracao que acertou Eucalipto1..5 em numero de folhas.

Arquivo funcional: scripts_cv/05_numero_folhas.py

Resultados no CSV /tmp/resultado_folhas_ajuste.csv:

Img,Altura Vert.,Compr Total,Diametro,Area,Nro Folhas
Eucalipto1,758,695,12,61644,11
Eucalipto2,1158,1009,18,209885,12
Eucalipto3,1098,1336,19,187933,12
Eucalipto4,794,626,16,45177,13
Eucalipto5,264,74,14,27868,4
Eucalipto6,148,50,0,17493,2
Eucalipto7,1078,825,16,76278,19
Eucalipto8,985,888,21,82983,12
Eucalipto9,1287,1088,17,122173,13
Eucalipto10,874,862,18,117199,8

MAPE Eucalipto1..5:
- Altura Vert.: 1.25%
- Compr Total: 1.51%
- Diametro: 8.31%
- Area: 2.87%
- Nro Folhas: 0.00%

Parametros principais da contagem:
- AREA_MIN_FOLHA_COUNT = 250
- AREA_MIN_FOLHA_FRACA = 150
- AREA_MUDA_PEQUENA = 35000
- AREA_MUDA_MEDIA = 80000
- AREA_MUDA_GRANDE = 100000
- COUNT_MIN_CONFIAVEL = 10
- RAZAO_COMPONENTE_DOMINANTE = 0.75

Regras:
- base: abertura eliptica 3x3 e componentes >= 250 px.
- se contagem base >= 10: usa base.
- se area <= 35000 e contagem <= 5: abertura 15x15, componentes >= 250 px.
- se area >= 100000 e maior componente >= 75% da area: abertura 13x13, componentes >= 250 px.
- se area <= 80000 e contagem < 10: abertura 9x9, componentes >= 150 px.
