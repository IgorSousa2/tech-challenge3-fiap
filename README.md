# Assistente Médico Inteligente com Fine-Tuning (LoRA) + LangChain/LangGraph

## Descrição

Este projeto implementa um assistente inteligente de apoio à decisão clínica, combinando **fine-tuning de um LLM open-source (Llama-3-8B) via LoRA/Unsloth** com um **agente orquestrado em LangChain + LangGraph**, que utiliza RAG (Retrieval-Augmented Generation) sobre protocolos internos do hospital e uma base simulada de prontuários.

O sistema é dividido em dois notebooks principais:

- **`tech_challenge_3_fine_tuning.ipynb`** — prepara os dados (dataset público `healthqa-br` + laudos internos simulados), realiza anonimização e curadoria, e executa o fine-tuning do modelo `unsloth/llama-3-8b-bnb-4bit` com LoRA.
- **`langgraph_langchain_assistente_medico.ipynb`** — carrega o modelo fine-tuned (adaptador LoRA), monta um índice RAG com os protocolos internos (PDFs) e executa um grafo de agente (LangGraph) que interpreta a pergunta do médico, consulta o prontuário do paciente, recupera o protocolo relevante e gera uma sugestão de conduta, sempre com validação de segurança e disclaimer de revisão médica obrigatória.

## Estrutura do Projeto

```
.
├── tech_challenge_3_fine_tuning.ipynb
├── langgraph_langchain_assistente_medico.ipynb
├── laudos-fraturas.xlsx
├── protocolos/
│   └── *.pdf
├── api-key.txt
├── symptom_x_diagnosis.json
├── diagnosis_dataset_chat_data.json              (gerado)
├── formatted_diagnosis_dataset_chat_data.json    (gerado)
└── lora_model/                                   (gerado — adaptador LoRA + tokenizer)
```

> **Observação:** todos os arquivos e datasets utilizados pelos notebooks (`laudos-fraturas.xlsx`, a pasta `protocolos/` com os PDFs, `api-key.txt` e `symptom_x_diagnosis.json`) já estão disponíveis na **raiz deste repositório**. Os arquivos gerados pelo próprio fine-tuning (`diagnosis_dataset_chat_data.json`, `formatted_diagnosis_dataset_chat_data.json` e a pasta `lora_model/`) são criados automaticamente durante a execução.

## Requisitos

- Conta Google (para uso do Google Colab)
- **GPU** — recomendado (ver seção abaixo)
- Chave de API da OpenAI (usada na etapa de sumarização do dataset público, dentro do notebook de fine-tuning)

As dependências (LangChain, LangGraph, Unsloth, PEFT, bitsandbytes, transformers, spaCy, sentence-transformers, faiss-cpu etc.) são instaladas diretamente nas primeiras células de cada notebook via `!pip install`, portanto não é necessário um `requirements.txt` separado.

## Execução com GPU

Ambos os notebooks foram desenvolvidos para rodar no **Google Colab** e dependem de bibliotecas que exigem GPU com suporte a CUDA:

- O fine-tuning usa **Unsloth**, **bitsandbytes** (quantização em 4-bit) e **PEFT/LoRA**, que precisam de GPU para treinar o modelo em tempo viável.
- O notebook do assistente também carrega o mesmo modelo base em 4-bit para gerar as respostas com o adaptador treinado.

Por isso, ao abrir os notebooks no Colab, selecione um ambiente de execução com GPU em **Ambiente de execução → Alterar tipo de ambiente de execução → GPU** (ex.: T4) antes de rodar as células.

Caso o notebook do assistente (`langgraph_langchain_assistente_medico.ipynb`) seja executado **sem GPU** ou sem o adaptador LoRA disponível, o carregamento do modelo falha de forma controlada: a variável `llm` permanece `None` e o grafo passa a responder em um **modo offline/simulado**, apenas para fins didáticos — ou seja, o notebook continua executável, mas sem as respostas geradas pelo modelo fine-tuned.

## Configuração dos caminhos do Google Drive

Os notebooks foram construídos assumindo que os arquivos estão salvos no Google Drive do autor original, na pasta `/content/drive/MyDrive/Tech_Challenge_3/`. Antes de rodar, **é necessário ajustar esses caminhos** para refletir onde os arquivos estão na sua própria conta:

- Ou copie os arquivos deste repositório (raiz) para uma pasta chamada `Tech_Challenge_3` na raiz do seu Google Drive, mantendo os caminhos padrão já usados no notebook; **ou**
- Altere manualmente cada variável/caminho que referencia `/content/drive/MyDrive/Tech_Challenge_3/...` (por exemplo `LORA_PATH`, `PROTOCOLOS_DIR`, `DATA_PATH`, `OUTPUT_PATH_DATASET`, o caminho do `laudos-fraturas.xlsx`, do `api-key.txt`, do `symptom_x_diagnosis.json` e do arquivo de log de auditoria) para a pasta do seu Drive onde os demais arquivos estiverem localizados.

Isso vale para os dois notebooks, já que ambos montam o Google Drive (`drive.mount('/content/drive')`) e leem/gravam arquivos a partir dele.

## Execução

### 1. Notebook de Fine-Tuning (`tech_challenge_3_fine_tuning.ipynb`)

1. Abra o notebook no Google Colab.
2. Selecione um ambiente de execução com GPU (ver seção acima).
3. Execute a célula de montagem do Google Drive e autorize o acesso.
4. Ajuste os caminhos do Drive conforme a seção anterior, garantindo que `laudos-fraturas.xlsx`, a pasta `protocolos/` e `symptom_x_diagnosis.json` estejam acessíveis nos caminhos referenciados.
5. Configure sua chave da OpenAI no arquivo `api-key.txt` (usado via `load_dotenv`).
6. Execute as células em ordem: instalação de dependências → carregamento e sumarização do dataset público (`healthqa-br`) → carregamento e anonimização dos laudos internos (Excel) → curadoria e consolidação do dataset → carregamento do modelo base em 4-bit → aplicação do LoRA → treinamento (`SFTTrainer`) → validação com exemplos de inferência.
7. Ao final, o adaptador LoRA e o tokenizer treinados são salvos na pasta `lora_model/` dentro do seu Drive — esse é o artefato usado pelo segundo notebook.

### 2. Notebook do Assistente (`langgraph_langchain_assistente_medico.ipynb`)

1. Abra o notebook no Google Colab (preferencialmente com GPU, para carregar o modelo fine-tuned).
2. Execute a montagem do Google Drive.
3. Ajuste os caminhos (`LORA_PATH`, `PROTOCOLOS_DIR`, arquivo de log de auditoria) para a pasta correta do seu Drive.
4. Execute as células em ordem: instalação de dependências → carregamento do LLM fine-tuned (ou fallback offline, se não houver GPU/adaptador) → montagem do índice RAG a partir dos PDFs de protocolos → criação da base simulada de prontuários (SQLite em memória) → definição do estado, dos nós e do grafo (LangGraph) → montagem do grafo → execução dos casos de exemplo ao final do notebook.
5. Os exemplos ao final simulam três cenários: uma dúvida clínica de rotina, um paciente com exame pendente (gera alerta) e uma pergunta fora do escopo clínico (rota de recusa).

## Vídeo de Demonstração

A demonstração completa do funcionamento deste projeto está disponível em:

https://youtu.be/CtX37T3m7zo

## Observações

- O sistema tem finalidade acadêmica e de demonstração (Tech Challenge 3).
- As sugestões geradas pelo assistente **não substituem avaliação médica profissional** — toda resposta é acompanhada de um disclaimer de validação humana obrigatória, e o grafo sinaliza automaticamente quando a resposta contém indícios de prescrição direta (dosagem, via de administração etc.).
- Os dados de pacientes e laudos utilizados são simulados/anonimizados; nenhuma informação real de paciente é utilizada.

## Autor
- Igor de Sousa
- RM 371788
- FIAP - Pós-Tech IA para Devs - 9IADT
- Grupo 92
