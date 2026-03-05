# Classificação de comportamento

Dashboard interativo de machine learning para classificar comportamentos de seis vacas da raça *Japanese Black* a partir de dados de acelerômetro tri-axial.

O projeto utiliza o conjunto de dados Japanese Black Beef Cow Behavior Dataset. A ideia é testar algumas etapas comuns em projetos de ciência de dados, como carregamento dos dados, extração de atributos, visualização, classificação com Random Forest e validação Leave-One-Cow-Out.

## Conjunto de dados

Conjunto de dados: [Japanese Black Beef Cow Behavior Dataset - Zenodo 5849025](https://zenodo.org/records/5849025)

Os arquivos CSV não estão armazenados neste repositório. Eles devem ser baixados diretamente da fonte original.

Para executar o projeto:

1. Acesse o registro no Zenodo pelo link acima.
2. Baixe o arquivo `JapaneseBlackBeefData.zip`.
3. Extraia os seis arquivos CSV para a pasta `data/`:
   - `cow1.csv`
   - `cow2.csv`
   - `cow3.csv`
   - `cow4.csv`
   - `cow5.csv`
   - `cow6.csv`

## Resumo do conjunto de dados

- 6 vacas da raça *Japanese Black*
- Acelerômetro tri-axial fixado no pescoço
- Frequência de amostragem: 25 Hz
- 13 classes de comportamento rotuladas
- Aproximadamente 197 minutos de dados rotulados

## Funcionalidades

- Carregamento dos dados por animal
- Visualização dos sinais de acelerômetro
- Extração de atributos com janela deslizante
- Cálculo de características como média, variação e padrões de movimento
- Visualização com PCA e t-SNE
- Classificação com Random Forest
- Validação Leave-One-Cow-Out
- Dashboard em Streamlit com filtros interativos
- Modo de exportação de gráficos para apresentações ou relatórios

## Estrutura do projeto

```text
.
├── app/
│   └── app.py
├── data/
│   ├── .gitkeep
│   └── ReadMe.md
├── notebooks/
│   └── 01_eda.ipynb
├── requirements.txt
└── README.md
```

## Instalação

```bash
pip install -r requirements.txt
```

## Como executar o dashboard

```bash
streamlit run app/app.py
```

## Métodos

A série temporal do acelerômetro é dividida em janelas deslizantes. Para cada janela, o projeto extrai estatísticas descritivas e atributos no domínio da frequência a partir dos eixos `AccX`, `AccY` e `AccZ`.

A classificação é feita com Random Forest. A validação Leave-One-Cow-Out é usada para avaliar o desempenho do modelo quando uma vaca fica fora do treinamento e é usada apenas no teste.

## Status do projeto

Este projeto ainda está em desenvolvimento e faz parte do meu aprendizado em ciência de dados aplicada à ciência animal.

Estou estudando machine learning com foco em aplicações para análise de dados na ciência animal, então sugestões, correções e colaborações são bem-vindas.

## Citação

Se utilizar este conjunto de dados, cite o dataset original e as publicações relacionadas indicadas no Zenodo:

[https://zenodo.org/records/5849025](https://zenodo.org/records/5849025)

## Licença

Verifique a licença e as restrições de uso do conjunto de dados no Zenodo antes de reutilizar ou redistribuir os dados. Este repositório não redistribui os arquivos CSV originais.
