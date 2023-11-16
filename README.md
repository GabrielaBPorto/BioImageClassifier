# Classificação de Imagens de Imunohistoquímica

## Descrição
Este projeto tem como objetivo classificar imagens de imunohistoquímica em quatro categorias (0, 1+, 2+, 3+) usando várias técnicas de processamento de imagem e aprendizado de máquina. O projeto envolve recorte de imagem, limiarização, extração de características com PyRadiomics e classificação usando classificadores KNN ou MLP da biblioteca scikit-learn.

## Estrutura de Pastas

- **data/**: Contém todos os dados usados e gerados no projeto.
    - **raw/input/imagens_ihq_er/**: Imagens de imunohistoquímica brutas organizadas em subdiretórios (0, 1, 2, 3) para cada classe.
    - **processed/**: Arquivos de dados processados.
        - **cropped_images/**: Armazena as imagens recortadas (40x30) derivadas das imagens originais.
        - **folded_data/**: Dados dobrados para validação cruzada, garantindo a separação por paciente.
    - **masks/**: Máscaras de limiarização usadas para extração de características.

- **docs/**: Documentação e relatórios relacionados ao projeto.
    - **relatorio.pdf**: Um relatório abrangente detalhando as metodologias, resultados e comparações de diferentes técnicas de limiarização.

- **notebooks/**: Notebooks Jupyter para exploração de dados e análise preliminar.
    - **exploration.ipynb**: Um exemplo de notebook para exploração e visualização inicial de dados.

- **src/**: Código-fonte para o projeto.
    - **preprocessing.py**: Código para pré-processamento de imagem, incluindo recorte.
    - **feature_extraction.py**: Extração de características usando PyRadiomics.
    - **classification.py**: Implementação de algoritmos de classificação.
    - **utilities.py**: Funções de utilidade usadas em todo o projeto.

- **tests/**: Casos de teste para o código do projeto.
    - **test_preprocessing.py**: Script de teste para as etapas de pré-processamento.

- **requirements.txt**: Lista todas as dependências do Python necessárias para o projeto.

- **.gitignore**: Especifica arquivos intencionalmente não rastreados para ignorar.

- **README.md**: Este arquivo, fornecendo uma visão geral do projeto e sua estrutura.
