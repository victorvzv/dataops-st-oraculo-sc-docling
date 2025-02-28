import streamlit as st
import os
import glob
import time
from typing import Iterator, Iterable, List, Set
from dotenv import load_dotenv
load_dotenv()

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


class DoclingLoader(BaseLoader):
    """
    Loader genérico que utiliza o Docling para converter arquivos de diversos formatos.
    """
    def __init__(self, file_paths: list[str]) -> None:
        self._file_paths = file_paths
        self._converter = DocumentConverter()

    def lazy_load(self) -> Iterator[LCDocument]:
        for source in self._file_paths:
            if source.lower().endswith(".txt"):
                with open(source, "r", encoding="utf-8") as f:
                    text = f.read()
                yield LCDocument(page_content=text, metadata={"source": source})
            else:
                try:
                    dl_doc = self._converter.convert(source).document
                    text = dl_doc.export_to_markdown()
                    yield LCDocument(page_content=text, metadata={"source": source})
                except Exception as e:
                    st.error(f"Erro ao processar {source}: {str(e)}")


def load_files_from_directory(directory: str, extensions: list[str] = None) -> list[str]:
    """
    Procura e retorna arquivos do diretório com as extensões especificadas.
    """
    if extensions is None:
        extensions = ['*.pdf', '*.docx', '*.txt', '*.xlsx', '*.csv', '*.json', '*.html', '*.xml', '*.md', '*.pptx', '*.odt']
 
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return files


def get_processed_files(log_file: str) -> Set[str]:
    """
    Lê o arquivo de log e retorna o conjunto de arquivos já processados.
    """
    if not os.path.exists(log_file):
        return set()
    
    with open(log_file, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def update_processed_files(log_file: str, new_files: List[str]) -> None:
    """
    Atualiza o arquivo de log com novos arquivos processados.
    """
    with open(log_file, "a", encoding="utf-8") as f:
        for file_path in new_files:
            f.write(f"{file_path}\n")


def format_docs(docs: Iterable[LCDocument]) -> str:
    """
    Formata os documentos em uma única string, separando cada conteúdo com duas quebras de linha.
    """
    return "\n\n".join(doc.page_content for doc in docs)


# Configuração da página Streamlit
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("📚 RAG Chatbot com Docling")
st.markdown("Pergunte qualquer coisa sobre os documentos carregados!")

# Inicializa o histórico de chat no estado da sessão
if "messages" not in st.session_state:
    st.session_state.messages = []

# Função para inicializar o RAG com processamento incremental
def initialize_rag():
    # Diretório atual onde estão os documentos
    current_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rags")
    
    # Verifica se o diretório existe
    if not os.path.exists(current_directory):
        os.makedirs(current_directory)
        st.warning(f"O diretório 'rags' foi criado. Por favor, adicione seus documentos nele.")
        st.stop()
    
    # Carrega lista de arquivos disponíveis
    file_paths = load_files_from_directory(current_directory)
    if not file_paths:
        st.warning("Nenhum arquivo encontrado no diretório 'rags'. Por favor, adicione documentos para consulta.")
        st.stop()
    
    # Arquivo para rastrear documentos já processados
    processed_files_log = "processed_files.txt"
    
    # Configurações para o banco de dados
    persist_directory = "chroma_db"
    collection_name = "first_rag"
    
    # Configura OpenAI API
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("Por favor, defina a variável de ambiente OPENAI_API_KEY.")
        st.stop()
    
    # Inicializa o modelo de embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=openai_api_key
    )
    
    # Verifica se existe um banco de dados persistido
    db_exists = os.path.exists(persist_directory) and os.listdir(persist_directory)
    
    # Progress bar para acompanhar o processo
    progress_bar = st.progress(0)
    
    if db_exists:
        # Carrega lista de arquivos já processados
        processed_files = get_processed_files(processed_files_log)
        
        # Identifica novos arquivos
        new_files = [f for f in file_paths if f not in processed_files]
        
        if new_files:
            st.info(f"Processando {len(new_files)} novos documentos...")
            
            # Carrega o banco existente
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                collection_name=collection_name
            )
            
            # Processa apenas os novos documentos
            loader = DoclingLoader(file_paths=new_files)
            new_docs = list(loader.lazy_load())
            
            if new_docs:
                # Atualiza a barra de progresso
                for i in range(3):
                    progress_bar.progress((i + 1) / 3)
                    time.sleep(0.1)
                
                # Divide os novos documentos
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2500,
                    chunk_overlap=200,
                )
                new_splits = text_splitter.split_documents(new_docs)
                
                # Adiciona novos documentos ao banco existente
                vectorstore.add_documents(new_splits)
                
                # Atualiza o registro de arquivos processados
                update_processed_files(processed_files_log, new_files)
                
                st.success(f"{len(new_files)} novos documentos adicionados ao banco de dados!")
            else:
                st.warning("Não foi possível extrair conteúdo dos novos arquivos. Verifique se os formatos são suportados.")
        else:
            st.success("Banco de dados atualizado - nenhum novo documento para processar")
            
            # Carrega o banco existente
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                collection_name=collection_name
            )
            
            # Atualiza a barra de progresso
            progress_bar.progress(1.0)
    else:
        # Primeira execução - processa todos os arquivos
        st.info(f"Criando novo banco de dados com {len(file_paths)} documentos...")
        
        # Processa todos os documentos pela primeira vez
        loader = DoclingLoader(file_paths=file_paths)
        docs = list(loader.lazy_load())
        
        if not docs:
            st.error("Não foi possível extrair conteúdo dos arquivos. Verifique se os formatos são suportados.")
            st.stop()
        
        # Atualiza a barra de progresso
        for i in range(3):
            progress_bar.progress((i + 1) / 3)
            time.sleep(0.1)
        
        # Divide os documentos
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,
            chunk_overlap=200,
        )
        splits = text_splitter.split_documents(docs)
        
        # Cria novo banco de dados
        vectorstore = Chroma.from_documents(
            splits,
            embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        
        # Registra os arquivos processados
        update_processed_files(processed_files_log, file_paths)
        
        st.success(f"Banco de dados criado com {len(file_paths)} documentos!")
        
        # Completa a barra de progresso
        progress_bar.progress(1.0)
    
    # Remove a barra de progresso após conclusão
    progress_bar.empty()
    
    # Configura o LLM
    llm = ChatOpenAI(
        model="o1-mini",
        max_retries=2,
        stream_usage=True,  
        openai_api_key=openai_api_key
    )
    
    # Configura o retriever
    retriever = vectorstore.as_retriever()
    
    # Template do prompt
    prompt = PromptTemplate.from_template(
        "# System Prompt\n"
        "Você é uma vendedora experiente e atenciosa da Disparo Pro, que se chama Jenifer. Seu objetivo é ajudar os clientes a entender melhor como funciona nosso produto, tirar dúvidas de como usar, como fazer recargas e principalmente fazer vendas efetivas através de uma jornada onde vc irá enctar o cliente com técnicas de vendas avançadas"
        "baseando-se nas informações disponíveis sobre os produtos e serviços da empresa. "
        "Comporte-se das seguintes formas:\n"
        "- Seja um pouco gentil, um pouco atenciosa, e com bom humor, para com o cliente\n"
        "- Após 10 perguntas do cliente a conversa deve se encerrar"
        "- Identifique o perfil do cliente durante a conversa e adapte sua abordagem de acordo rapidamente\n"
        "- NUNCA fale mal da Disparo Pro, você é um vendedor leal\n"
        "- Apenas se apresente como Jenifer no início da conversa, depois você não deve mais apresentar seu nome, somente se o cliente pedir\n"
        "- Jamais invente informações ou dados que não sejam reais à respeito da Disparo Pro\n"
        "- Quando não souber alguma informação diga que não sabe e peça para o cliente entrar em contato com o Suporte ao cliente no seguinte número de WhatsApp: 551142994000, de Segunda-feira à Sexta-feira, nos horários 08:00 até 18:00, GMT -3 \n"
        "- Use técnicas eficazes de vendas e gatilhos emocionais quando apropriado\n"
        "- Responda como um humano, mimetize conversas como os seres humanos fazem entre si, evite linguagem robótica ou muito formal\n"
        "- Se questionada sobre assuntos não relacionados aos produtos da Disparo Pro, explique educadamente que você é uma vendedora especializado da empresa\n"
        "- Tente ao máximo parecer uma vendedora humana, com traços e comportamento humano, não uma inteligência artificial\n"
        "- Se o cliente perguntar em qual língua você consegue se comunicar, responda que você consegue conversar em qualquer língua\n"
        "# Contexto de Informações\n"
        "---------------------\n"
        "{context}\n"
        "---------------------\n"
        "Baseado no contexto acima e no seu papel como vendedor profissional da Disparo Pro, responda às perguntas dos clientes de forma persuasiva e natural.\n"
        "Pergunta do cliente: {question}\n"
        "Responde ao cliente conforme a língua qual ele estiver conversando:\n"
    )
    
    # Cria a cadeia RAG
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, file_paths


# Inicializa a barra lateral
st.sidebar.title("Informações")

# Inicializa o RAG
with st.spinner("Inicializando o sistema..."):
    rag_chain, file_paths = initialize_rag()

# Exibe informações sobre documentos processados
st.sidebar.subheader("Documentos Disponíveis")
st.sidebar.info(f"Total de {len(file_paths)} documentos carregados")

# Opcionalmente, mostra a lista de arquivos em um expander na barra lateral
with st.sidebar.expander("Ver lista de documentos"):
    for file in file_paths:
        st.write(f"- {os.path.basename(file)}")

# Exibe mensagens anteriores do histórico
for message in st.session_state.messages:
    with st.chat_message("user" if message["role"] == "user" else "assistant"):
        st.markdown(message["content"])

# Entrada do usuário
if prompt := st.chat_input("Digite sua pergunta..."):
    # Adiciona a mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Exibe a mensagem do usuário
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Gera resposta
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Mostra um indicador de carregamento enquanto processa
        with st.spinner("Processando sua pergunta..."):
            response = rag_chain.invoke(prompt)
        
        # Exibe a resposta
        message_placeholder.markdown(response)
    
    # Adiciona a resposta ao histórico
    st.session_state.messages.append({"role": "assistant", "content": response})

# Adiciona um botão para limpar o histórico
if st.sidebar.button("Limpar conversa"):
    st.session_state.messages = []
    st.experimental_rerun()

# Adiciona um botão para reprocessar todos os documentos
if st.sidebar.button("Reprocessar todos os documentos"):
    # Remove o arquivo de log para forçar o reprocessamento
    if os.path.exists("processed_files.txt"):
        os.remove("processed_files.txt")
    st.sidebar.success("Na próxima execução, todos os documentos serão reprocessados")
    st.experimental_rerun()
    # cuidar aqui pq teria que pensar onde esse arquivo seria armazenado