# Vetorização e balanceamento das bases

Este diretório contém o pipeline de pré-processamento das bases de dados utilizadas para o treinamento das instâncias do classificador. 

O fluxo de preparação de cada base de imagens consiste primordialmente em duas etapas:
- **Vetorização com FaceNet**
- **Balanceamento de Classes** (via *Oversampling* com SMOTE ou *Undersampling* genérico por exclusão de excedentes)
<p><br></p>

---

## Bases de imagens utilizadas

Foram utilizadas duas bases de iamgens distintas, configuradas conforme a tabela abaixo:

| Base de Dados | Descrição | Quantidade de Imagens |
| :--- | :--- | :--- |
| [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | Composto por mais de 200.000 imagens anotadas automaticamente com 40 atributos binários (ex: `young`, `have_mustache`, `male`). | 202.000 imagens selecionadas do total disponível. |
| [Casual Conversations v2](https://ai.meta.com/datasets/casual-conversations-v2-dataset/) | Composto por mais de 26.000 vídeos divididos em frames, contendo atributos autoanotados (ex: `fitzpatrick_skintone`, `monk_scale`, `hair_type`, `hair_color`, `eye_color`). | 230.000 frames extraídos do total disponível. |
<p><br></p>

---

## Vetorização com FaceNet

É a primeira etapa do pré-processamento, e é nela em que o script Python `loader_{dataset_name}.py` percorre os diretórios das bases e gera os *embeddings* processados em lotes (*batches*). 

Para cada lote, a extração de características segue rigorosamente a seguinte ordem de execução:
1º) **Detecção Facial:** Identificação e isolamento da face na imagem (quando detectada).
2º) **Recorte (*Cropping*):** Redimensionamento e recorte da região exata da face identificada.
3º) **Vetorização e Concatenação:** Geração dos vetores de características face a face, concatenando-os diretamente com suas respectivas anotações binárias/categóricas.
<p><br></p>

---

## 2. Balanceamento das bases de imagens

Após a vetorização completa, o balanceamento é realizado com base na distribuição de classes do atributo mais relevante de cada base. O script `balancing_{dataset_name}.py` executa esse processo seguindo os critérios abaixo:

* **Seleção do Pivô:** Escolha do atributo principal que servirá de referência para o balanceamento da base.
* **Definição da Estratégia:**
   * **Oversampling:** Aplicado ao atributo `fitz_type` na base *Casual Conversations v2*.
   * **Undersampling:** Aplicado ao atributo `Male` na base *CelebA Dataset*.
* **Aplicação do Método:**
   * Para o cenário de *oversampling*, utilizou-se o algoritmo **SMOTE**.
   * Para o cenário de *undersampling*, aplicou-se a **exclusão aleatória dos dados excedentes** da classe majoritária.
<p><br></p>

---

## Fluxo Visual de Execução

```mermaid
graph TD
    A[Base de Imagens Bruta] --> B[loader_.py]
    B --> C{Face Detectada?}
    C -- Sim --> D[Recorte da Face]
    C -- Não --> E[Descarte/Próxima Imagem]
    D --> F[Extração de Embeddings - FaceNet]
    F --> G[Concatenação: Vetor + Anotações]
    G --> H[balancing_.py]
    H --> I{Estratégia Escolhida}
    I -- Casual Conversations --> J[Oversampling: SMOTE]
    I -- CelebA --> K[Undersampling: Exclusão Arbitrária]
    J --> L[Base Pronta para Treinamento]
    K --> L
```
