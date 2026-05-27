# classificador-massivo-de-atributos

<p>
  Esse repositório implementa fielmente o classificador de atributos faciais massivo idealizado por Therhörst,<br>
  em <a href="https://arxiv.org/abs/2012.01030">MAAD-Face: A Massive Annotated Attribute Dataset for face images</a>
</p>

---

### Fluxo do código
<p>
  O classificador foi implementado seguindo, em ordem, 3 processos bem definidos:<br>
<table>
  <tr><td>1º processo</td><td>Preparação dos datasets (dados crus).</td></tr>
  <tr><td>2º processo</td><td>Ingestão dos dados e definição da rede neural do modelo.</td></tr>
  <tr><td>3º processo</td><td>Anotação de um dataset inteiro com a função de predição resultante.</td></tr>
</table>
</p>

### Os datasets utilizados para treinamento do modelo
<p>
  Utilizou-se 2 bases de dados diferentes, que são definidas a seguir:<br>
  <table>
    <tr><td><strong>Base de dados</strong></td><td><strong>Descrição</strong></td><td><strong>Quantidade de imagens</strong></td></tr>
    <tr><td><a href="https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html">CelebA Dataset</a></td>
      <td>Composto por mais de 200.000 imagens anotadas automaticamente com 40 atributos binários. Dentre eles:
      young, have_mustache, male</td><td>Utilizou-se 202.000 imagens, do total disponível</td></tr>
    <tr><td><a href="https://ai.meta.com/datasets/casual-conversations-v2-dataset/">Casual Conversations v2</a></td>
      <td>Composto por mais de 26.000 vídeos, divididos em frames, com atributos autoanotados. Dentre eles:
      fitzpatrick_skintone, monk_scale, hair_type, hair_color e eye_color</td><td>Utilizou-se 230.000 imagens, do total de frames disponíveis</td></tr>
  </table>
</p>

### A definição formal da rede neural que foi implementada
