## data_prep
<p>
  Nesta pasta, realizei o pré-processamento das base de dados que usei para treinar o classificador massivo.
</p>

<p><br></p>

### As bases de imagens utilizadas
<p>
  Utilizei 2 bases de imagens diferentes, que são definidas a seguir:<br>
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

<p><br></p>

### O pré-processamento
<p>As etapas de pré-processamento de cada uma das bases de imagens segue a seguinte ordem:<br>
<b>1º)</b> Vetorização via FaceNet. O resultado desta etapa é um DataFrame Pandas, em formato .pkl, com os embeddings e os seus respectivos atributos pré-anotados.<br>
- proc/proc_celeba/embeddings_celeba.pkl<br>
- proc/proc_casual/embeddings_casual.pkl<br>
<b>2º)</b> Balanceamento das bases de imagens já vetorizadas com base em um dos atributos. O algoritmo de oversampling escolhido foi o SMOTE.
</p>


### 1º) Vetorização via FaceNet
<p>
Para esta etapa, o script python <b><i>loader_{dataset_name}.py</i></b>, percorre o caminho de pastas da base de imagens e gera embeddings por batchs de imagens.<br>
Para gerar os batchs de embeddings, cada batch de imagem:<br>
<ul>
	<li><b>- Extrai a face, caso seja detectada</b></li>
	<li><b>- Recorta a face detectada</b></li>
	<li><b>- Vetoriza face a face, concatenando-a com sua respectiva anotação</b></li>
</ul>
</p>


### 2º) Balanceamento das bases de imagens
<p>
Para cada base de imagem vetorizada, escolhemos o atributo mais relevante e balanceamos todo a base de acordo com a sua distribuição.<br>
Para isso:<br>
<ul>
	<li><b>- Definimos o atributo que será utilizado como pivô para o balanceamento da base de dados inteira
	<li><b>- Decide-se se o princípio de balanceamento será por <i>oversampling</i> ou <i>undersampling</i>. Decidimos por oversampling para o casual e undersampling para o celeba</b></li>
	<li><b>- Por fim, definimos o método de oversampling ou undersampling. Neste caso, o método de oversampling escolhido foi o SMOTE
</ul>
O script python <b><i>balancing_{dataset_name}.py</i></b>realiza o balanceamento das suas respectivas bases de imagem.
</p>

### Fluxo visual de execução
