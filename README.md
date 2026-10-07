# Projeto-de-PLN-Healthtec

Mapeamento Topológico e Classificação Semântica de Literatura Médica
Este repositório contém a implementação de um pipeline completo de Processamento de Linguagem Natural (NLP) voltado à classificação hierárquica de textos no domínio biomédico. O projeto tem como foco central a equidade algorítmica (algorithmic fairness), aplicando técnicas estritas de undersampling para mitigar o viés de representatividade (volume) de classes majoritárias. Esta abordagem garante que a identificação de especialidades de nicho e doenças raras possua o mesmo peso preditivo que patologias epidemiologicamente mais frequentes.

O presente trabalho foi desenvolvido como requisito avaliativo para a disciplina de Tópicos em Biotecnologia 2, do Programa de Pós-Graduação em Biotecnologia (PPGBiotec) da Universidade Federal do Delta do Parnaíba (UFDPar).

Arquitetura do Modelo
O pipeline metodológico afasta-se de abordagens fundamentadas exclusivamente na frequência de termos (Bag-of-Words), propondo uma arquitetura híbrida que integra espaços vetoriais densos e análise de redes complexas:

Vetorização Semântica (Embeddings): Conversão do corpus textual em vetores densos de 384 dimensões, utilizando o modelo Transformer all-MiniLM-L6-v2 (Sentence-BERT).

Macro-Áreas (Baseline Probabilístico): Emprego de Modelagem de Tópicos via Latent Dirichlet Allocation (LDA) para categorização macro-semântica primária.

Micro-Comunidades (Topologia de Grafos): Construção de um grafo K-Nearest Neighbors (KNN) a partir da similaridade cosseno dos embeddings, seguida da aplicação do algoritmo de detecção de comunidades de Louvain. Esta etapa permite o agrupamento não-supervisionado de nichos clínicos de alta especificidade (e.g., Oncologia Molecular, Nanotecnologia Farmacêutica, Genética Médica).

Classificação Preditiva: Treinamento de um modelo Support Vector Machine (LinearSVC) sobre o espaço dimensional reduzido e balanceado, visando inferência com baixo custo computacional em ambiente de produção.

Estrutura do Repositório

TRABALHO_NLP.ipynb: Notebook Jupyter contendo a implementação integral do pipeline (aquisição de dados via repositório remoto, geração de embeddings, aplicação de undersampling, treinamento dos modelos, extração de métricas de avaliação multivariada e interface para inferência).

modelo_svm_lda.pkl: Objeto serializado contendo os pesos do classificador SVM otimizado para a predição das 5 Macro-Áreas.

modelo_svm_louvain.pkl: Objeto serializado contendo os pesos do classificador SVM treinado para a predição das 20 Micro-Comunidades topológicas.

perfil_comunidades_louvain.pkl: Dicionário semântico serializado contendo a extração de features (palavras-chave via TF-IDF) que definem e rotulam dinamicamente cada micro-comunidade detectada.

Reprodutibilidade e Execução
Reprodução Integral do Pipeline
Importe o arquivo TRABALHO_NLP.ipynb em um ambiente Google Colab.

Selecione a opção Ambiente de Execução > Executar Tudo.

O script executará automaticamente a extração do conjunto de dados em formato Parquet, o particionamento estratificado, o treinamento dos modelos paramétricos e a plotagem das Matrizes de Confusão e Classification Reports.

Deploy e Validação Rápida (Inferência Interativa)
Para validar a precisão do modelo com novos dados textuais sem reprocessar as etapas de treinamento:

Abra o arquivo TRABALHO_NLP.ipynb no Google Colab.

Realize o upload dos três artefatos serializados (.pkl) no diretório virtual local do ambiente.

Execute a célula correspondente ao recarregamento das variáveis (instrução joblib.load()).

Navegue até o final do documento e execute o bloco correspondente à Interface Preditiva Dinâmica.

Utilize a interface gerada via ipywidgets para submeter resumos clínicos ou relatos médicos originais (idioma inglês) e inspecionar a ancoragem semântica processada pelo algoritmo em tempo real.

Stack Tecnológico
Linguagem: Python 3.10+

Vetorização e NLP: sentence-transformers (Hugging Face)

Aprendizado de Máquina: scikit-learn

Análise de Redes: networkx, python-louvain (community)

Engenharia de Dados: pandas, numpy

Visualização e Componentes Visuais: matplotlib, seaborn, ipywidgets
