# Import necessary libraries
import os  # For environment variables and file operations
import pandas as pd  # For data manipulation
from langchain_huggingface import HuggingFaceEmbeddings  # For text embeddings using HuggingFace models
from langchain_community.vectorstores import FAISS  # For vector storage and similarity search
from langchain_community.document_loaders import CSVLoader  # For loading CSV files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into chunks
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # For OpenAI's language models and embeddings
from langchain_pinecone import PineconeVectorStore  # For Pinecone vector database
from langchain.retrievers import EnsembleRetriever  # For combining multiple retrievers
from dotenv import load_dotenv  # For loading environment variables
from langchain.prompts import PromptTemplate  # For creating prompt templates
from langchain.memory import ConversationBufferMemory  # For maintaining conversation history
import re  # For regular expressions
import pinecone  # For Pinecone vector database operations
from langchain_community.tools.brave_search.tool import BraveSearch  # For web search functionality
import streamlit as st  # For creating web interface

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")  # Get OpenAI API key
pinecone_api_key = os.getenv("PINECONE_API_KEY")  # Get Pinecone API key
brave_api_key = os.getenv("BRAVE_API_KEY")  # Get Brave Search API key
llm_model = "gpt-4o"  # Specify the language model to use

# Initialize embeddings with caching to avoid PyTorch conflicts
@st.cache_resource
def get_embeddings(model: str):
    """
    Get text embeddings based on specified model
    Args:
        model (str): Either 'openai' or 'generic' for HuggingFace
    Returns:
        Embeddings object for the specified model
    """
    if model == "openai":
        return OpenAIEmbeddings(model = "text-embedding-3-small", api_key = "") #type: ignore
    else:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Cache vector store initialization to run only once
@st.cache_resource
def initialize_vector_stores(model:str, faiss_docs_dir:str):
    """
    Initialize and return vector stores (FAISS and Pinecone).
    If Pinecone index already exists, skip document ingestion.
    
    Args:
        model (str): Embedding model to use ('openai' or 'generic')
        faiss_docs_dir (str): Directory for FAISS index storage
        
    Returns:
        tuple: (faiss_store, pinecone_store) - The initialized vector stores
    """
    # Get embeddings based on model choice
    embeddings = get_embeddings(model)
    print("this is the embeddings", embeddings)
    if model == "openai":
        embedding_size = 1536  # OpenAI embedding dimension
    else:
        embedding_size = 384  # HuggingFace embedding dimension
    
    # Define data sources and categories
    file_paths = ["df_skin.csv", "df_hair.csv", "df_vits_supp.csv"]
    namespaces = {"df_skin.csv": "skin", "df_hair.csv": "hair", "df_vits_supp.csv": "vitamins_supplements"}
    faiss_docs = []
    pinecone_docs = {"skin": [], "hair": [], "vitamins_supplements": []}

    # Initialize Pinecone client
    pc = pinecone.Pinecone(api_key=pinecone_api_key)

    # Set index name based on model
    if model == "openai":
        index_name = "clinikally-rag-2"
    else:
        index_name = "clinikally-rag"
    index_exists = False

    # Check if Pinecone index exists
    try:
        indexes = pc.list_indexes()
        index_exists = any(index.name == index_name for index in indexes)
        
        if index_exists:
            print(f"Index '{index_name}' already exists. Skipping document ingestion.")
        else:
            print(f"Index '{index_name}' does not exist. Will create and ingest documents.")
            pc.create_index(index_name, dimension= embedding_size, spec = pinecone.ServerlessSpec(cloud="aws", region="us-east-1"))
    except Exception as e:
        print(f"Error checking Pinecone index: {e}")
        print("Will proceed with document processing for FAISS.")

    # Process documents for vector stores
    for file in file_paths:
        # Load CSV data
        loader = CSVLoader(file_path=file)
        documents = loader.load()
        
        # Extract metadata from documents
        for doc in documents:
            content_lines = doc.page_content.split('\n')
            title = ""
            price = ""
            
            # Parse document content for metadata
            for line in content_lines:
                if line.startswith("Title:"):
                    title = line.replace("Title:", "").strip()
                elif line.startswith("Variant Price:"):
                    price_str = line.replace("Variant Price:", "").strip()
                    try:
                        price = float(price_str)
                    except:
                        price = price_str
                elif line.startswith("Metafield: my_fields.brand_name [single_line_text_field]: "):
                    brand = line.replace("Metafield: my_fields.brand_name [single_line_text_field]: ", "").strip()
            
            # Add metadata to document
            doc.metadata["Title"] = title
            doc.metadata["Price"] = price
            doc.metadata["Brand"] = brand

        # Split documents into chunks for better processing
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = text_splitter.split_documents(documents)
        namespace = namespaces[file]
        
        # Add category metadata and prepare documents for stores
        for doc in split_docs:
            doc.metadata["Category"] = namespace
            faiss_docs.append(doc)
            
            if not index_exists:
                pinecone_docs[namespace].append(doc)
        
        print(f"Number of documents in {namespace}: {len(split_docs)}")
        print(f"Number of documents in Faiss: {len(faiss_docs)}")

    # Initialize FAISS store
    if os.path.exists(faiss_docs_dir) and faiss_docs_dir == "faiss_index":
        print(f"Loading OpenAI FAISS store from {faiss_docs_dir}")
        faiss_store = FAISS.load_local(faiss_docs_dir, embeddings, allow_dangerous_deserialization=True)   
    elif os.path.exists(faiss_docs_dir) and faiss_docs_dir == "faiss_index_generic":
        print(f"Loading Generic FAISS store from {faiss_docs_dir}")
        faiss_store = FAISS.load_local(faiss_docs_dir, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"Creating new FAISS store in {faiss_docs_dir}")
        faiss_store = FAISS.from_documents(faiss_docs, embeddings)
        faiss_store.save_local(faiss_docs_dir)

    # Initialize Pinecone stores for each category
    pinecone_store = {}
    for namespace in namespaces.values():
        vector_store = PineconeVectorStore(
            index_name=index_name, 
            embedding=embeddings, 
            text_key="text", 
            namespace=namespace
        )
        
        # Add documents if index is new
        if not index_exists and namespace in pinecone_docs and pinecone_docs[namespace]:
            print(f"Ingesting documents to Pinecone namespace: {namespace}")
            vector_store.add_documents(pinecone_docs[namespace])
        
        pinecone_store[namespace] = vector_store
    
    return faiss_store, pinecone_store

def classify_query(query):
    """
    Classifies a query as either product-based or general information.
    Uses LLM for classification with fallback to keyword-based heuristics.
    
    Args:
        query (str): The user's query
        
    Returns:
        str: Either "product" or "general"
    """
    # Initialize language model
    llm = ChatOpenAI(model=llm_model, api_key=openai_api_key) #type: ignore
    
    # Create classification prompt template
    classification_template = """
    Determine if the following query is asking about specific products to purchase or if it's asking for general information/advice.
    
    Query: {query}
    
    If the query is about finding, buying, or comparing specific products, respond with "product".
    If the query is asking for general information, advice, how-to guides, or explanations, respond with "general".
    
    Your response should be exactly one word, either "product" or "general".
    """
    
    classification_prompt = PromptTemplate(
        template=classification_template,
        input_variables=["query"]
    )
    
    # Create and run classification chain
    classification_chain = classification_prompt | llm
    result = classification_chain.invoke({"query": query}).content.strip().lower() #type: ignore
    
    # Fallback to keyword-based classification if LLM fails
    if result not in ["product", "general"]:
        product_keywords = ["product", "buy", "purchase", "recommend", "price", "cost", "rupees", "rs", "₹", "brand", "where to get", "suggest", "suggestions"]
        general_keywords = ["how to", "why", "what causes", "explain", "treatment", "remedy", "cure", "prevent", "tips", "advice"]
        
        product_score = sum(1 for keyword in product_keywords if keyword in query.lower())
        general_score = sum(1 for keyword in general_keywords if keyword in query.lower())
        
        result = "product" if product_score >= general_score else "general"
    
    print(f"Query classified as: {result}")
    return result

def select_retrievers(query, faiss_store, pinecone_store):
    """
    Selects appropriate retrievers based on query content and filters.
    Handles price, brand, and rating filters.
    
    Args:
        query (str): User query
        faiss_store: FAISS vector store
        pinecone_store: Pinecone vector store
        
    Returns:
        list: List of selected retrievers
    """
    selected_retrievers = []
    query_lower = query.lower()
    
    # Initialize filter dictionary
    filter_dict = {}
    
    # Extract price filters using regex
    under_match = re.search(r'under\s+(\d+(?:\.\d+)?)\s+(?:rs\.?|rupees?)', query_lower)
    if under_match:
        try:
            price_threshold = float(under_match.group(1))
            filter_dict["Price"] = {"$lt": price_threshold}
        except ValueError:
            pass
    
    less_than_match = re.search(r'less\s+than\s+(\d+(?:\.\d+)?)\s+(?:rs\.?|rupees?)', query_lower)
    if less_than_match and "Price" not in filter_dict:
        try:
            price_threshold = float(less_than_match.group(1))
            filter_dict["Price"] = {"$lt": price_threshold}
        except ValueError:
            pass
    
    between_match = re.search(r'between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s+(?:rs\.?|rupees?)', query_lower)
    if between_match:
        try:
            min_price = float(between_match.group(1))
            max_price = float(between_match.group(2))
            filter_dict["Price"] = {"$gte": min_price, "$lte": max_price}
        except ValueError:
            pass
    
    # List of available brands
    brands = ['Cadila Pharmaceuticals Limited', 'Barulab', 'IPCA', 'Verso', 'Maddox Biosciences', 'Catalysis S.L.', 'Aethicz Biolife', 'iS Clinical', 'Skinnovation Next', 'Medever Healthcare', 'Carbamide Forte', 'Cantabria Labs', 'KAINE', 'Glint Cosmetics', 'Ceuticoz', 'Wallace Pharmaceuticals', 'Torrent Pharmaceuticals', 'Dermawiz Laboratories', 'Geosmatic Cosmeceuticals & Cosmocare', 'Aveeno', 'Rexcin Pharmaceuticals', 'Aurel Derma', 'Sanosan', 'Beauty of Joseon', 'Ira Berry Creations', 'Galderma', 'Rene Furterer', 'bhave', 'MRHM Pharma', 'Linux Laboratories', 'EVE LOM', 'Curatio', 'Dermis Oracle', 'Fixderma', 'Mankind Pharma Ltd.', 'Ivatherm', 'Brillare', 'Dermaceutic', "Re'equil", 'USV Private Limited', 'Fillmed Laboratories', 'Regaliz', 'Gracederma Healthcare', 'East West Pharma', 'Neutrogena', 'Entod Pharmaceuticals', 'Dermx', 'Percos India', 'Velite India', 'HBC Dermiza Health Care', 'PHILIP B', 'RevitaLash', 'Klairs', 'Alkem Laboratories', 'Ajanta Pharma', 'O3+', 'Canixa Life Sciences', 'Epique', 'Abbott', 'Multiple Brands', 'Sol Derma Pharma', 'Fluence Pharma', 'La Pristine', 'Yuderma', 'Intas Pharmaceuticals', 'OZiva', "Burt's Bees", 'KAHI', 'Dermatica', 'Ora Pharmaceuticals', 'Swisse', 'Mustela', 'LISEN', 'The FormulaRx', 'WishNew Wellness', 'Talent India', 'Eterno Distributors', 'Alembic', 'Mohrish Pharmaceuticals', 'Meconus Healthcare', 'Akumentis Healthcare Ltd.', 'Aclaris Therapeutics', 'Some By Mi', 'Zydus Healthcare', 'Adonis Phytoceuticals', 'Aveil', 'Tricept Life Sciences', 'Trilogy', 'Craza Lifescience', 'Sun Pharma', 'Apex Laboratories', 'Mylan Pharmaceuticals', 'Genosys', 'Cellula', 'Dermajoint India', 'Uriage', 'Kshipra Health Solutions', 'Gufic Biosciences Limited', 'Skinska Pharmaceutica', 'GlaxoSmithKline Pharmaceuticals Ltd', 'Swiss Perfection', 'Azelia Healthcare', 'Sedge Bioceuticals', 'Ethiall Remedies', 'Cipla', 'KLM Laboratories', 'Wellbeing Nutrition', 'Biocon Biologics', "Dr. Reddy's Laboratories", 'Sesderma', 'Ethicare Remedies', 'Embryolisse', 'Cosmogen India', 'Skinmedis', 'AMA Herbal Laboratories', 'Syscutis Healthcare', 'ISDIN', 'Glenmark Pharmaceuticals', 'Adroit Biomed Ltd', 'COSRX', 'Belif', 'Encore Healthcare', 'Ultra V', 'Regium Pharmaceuticals', 'Dabur India', 'Glo Blanc', 'Elder Pharma', 'Bellacos Healthcare', 'Apple Therapeutics', 'Eris Oaknet', 'Velsorg Healthcare', 'Ralycos LLC', 'P & J Labs', 'Crystal Tomato', 'Elcon Drugs & Formulations', 'Leafon Enterprises', 'Beta Drugs Ltd.', 'Lupin Limited', 'Nourrir Pharma', 'Universal Nutriscience', 'Leeford Healthcare', 'CeraVe', 'IUNIK', 'Meghmani Lifesciences', 'Clinikally', 'Der-Joint Healthcare', 'Biopharmacieaa', 'Rockmed Pharma', 'Gunam', 'Kativa', 'Blistex', 'Aauris Healthcare', 'Cosderma Cosmoceuticals', 'ZO Skin Health', 'Palsons Derma', 'Tricos Dermatologics', 'Bioderma', 'Meyer Organics', 'Salve Pharmaceuticals', "D'Vacos Cosmetics & Neutraceuticals", 'INJA Wellness', 'Brinton Pharmaceuticals', 'Unimarck Pharma', 'Coola LLC', 'Leo Pharma', 'Ethinext Pharma', 'Nutracos Lifesciences', 'Noreva Laboratories', 'Saion International', 'One Thing', 'Quintessence Skin Science', 'Soteri Skin', 'Glowderma', 'Zofra Pharmaceuticals', 'Strenuous Healthcare', 'Nimvas Pharmaceuticals', 'General Medicine Therapeutics', 'La Med India', 'UAS Pharmaceuticals', 'Capeli', 'Biokent Healthcare', 'Iceberg Healthcare', 'Novology', 'WishCare', 'Emcure Pharmaceuticals', 'HK Vitals', 'Cosmofix Technovation', 'Wockhardt', 'Janssen', 'Senechio Pharma', 'Indiabulls Pharmaceuticals', 'Micro Labs', 'A-Derma', 'Mediste Pharmaceuticals', 'Trikona Pharmaceuticals', 'Awear Beauty', 'BABE Laboratorios', 'Indolands Pharma', 'La Roche-Posay', 'Protein Kera', 'Dermalogica', 'Sebamed', 'Skyntox', 'Hegde & Hegde Pharmaceutical LLP', 'Oaknet Healthcare', 'Torque Pharmaceuticals', 'Renewcell Cosmedica', 'Bayer Pharmaceuticals', 'Ultrasun', 'ISIS Pharma', 'Win-Medicare Pvt Ltd', 'Kemiq Lifesciences', 'The Face Shop','Kerastem', 'Beautywise', 'Justhuman', 'A. Menarini India', 'Nutrova', 'Iberia Skinbrands', 'Haruharu']
    
    # Check for brand mentions in query
    for brand in brands:
        if brand.lower() in query_lower:
            filter_dict["Brand"] = {"$eq": brand}
            break
    
    # Extract rating filters
    rating_above_match = re.search(r'rat(?:ing|ed)\s+(?:above|over)\s+(\d+(?:\.\d+)?)', query_lower)
    if rating_above_match:
        try:
            rating_threshold = float(rating_above_match.group(1))
            filter_dict["reviews.rating_count"] = {"$gt": rating_threshold}
        except ValueError:
            pass
    
    stars_match = re.search(r'(\d+(?:\.\d+)?)\s+stars?', query_lower)
    if stars_match and "reviews.rating_count" not in filter_dict:
        try:
            rating_threshold = float(stars_match.group(1))
            filter_dict["reviews.rating_count"] = {"$gte": rating_threshold}
        except ValueError:
            pass
    
    # Define category patterns for matching
    patterns = {
        "skin": r'\b(skin|face|acne|pimple|complexion|blemish|wrinkle|dark spot|dark circle|pore|blackhead|whitehead|rash|dermatitis|eczema|psoriasis|rosacea|pigmentation|scar|aging|moisturizer|cleanser|toner|serum|sunscreen|skincare)\b',
        "hair": r'\b(hair|scalp|dandruff|hairfall|hair loss|hair growth|split end|frizz|dry hair|oily hair|hair care|shampoo|conditioner|hair mask|hair treatment|hair color|hair dye|hair style|hair product|balding|thinning|alopecia|grey hair|hair texture|hair volume)\b',
        "vitamins_supplements": r'\b(vitamin|supplement|mineral|nutrition|nutrient|deficiency|dietary|multivitamin|antioxidant|omega|protein|calcium|iron|magnesium|zinc|potassium|biotin|collagen|probiotic|prebiotic|amino acid|herbal|natural supplement|wellness|immunity|energy|metabolism)\b'
    }
    
    # Match query against category patterns
    matched_categories = []
    for category, pattern in patterns.items():
        if re.search(pattern, query_lower):
            matched_categories.append(category)
    
    # Use all categories if no specific matches
    if not matched_categories:
        matched_categories = list(pinecone_store.keys())
    
    # Create retrievers for matched categories
    for category in matched_categories:
        if category in pinecone_store:
            try:
                if filter_dict:
                    selected_retrievers.append(pinecone_store[category].as_retriever(search_kwargs={"filter": filter_dict}))
                else:
                    selected_retrievers.append(pinecone_store[category].as_retriever())
            except Exception as e:
                print(f"Error creating retriever for {category}: {e}")
                selected_retrievers.append(pinecone_store[category].as_retriever())
    
    # Add FAISS retriever as fallback
    try:
        if filter_dict:
            selected_retrievers.append(faiss_store.as_retriever(search_kwargs={"filter": filter_dict}))
        else:
            selected_retrievers.append(faiss_store.as_retriever())
    except Exception as e:
        print(f"Error creating FAISS retriever: {e}")
        selected_retrievers.append(faiss_store.as_retriever())
    
    if filter_dict:
        print(f"Applying filters: {filter_dict}")
    
    return selected_retrievers

def get_memory():
    """
    Initialize and return a ConversationBufferMemory instance.
    Used to maintain conversation history.
    
    Returns:
        ConversationBufferMemory: The initialized memory
    """
    return ConversationBufferMemory(
        llm=ChatOpenAI(model=llm_model, api_key=openai_api_key),  # type: ignore
        return_messages=True
    )


# Initialize memory in Streamlit session state
if "memory" not in st.session_state:
    st.session_state.memory = get_memory()
memory = st.session_state.memory

def process_query(query, faiss_store, pinecone_store):
    """
    Main function to process user queries.
    Routes queries to appropriate handlers based on classification.
    
    Args:
        query (str): User's query
        faiss_store: FAISS vector store
        pinecone_store: Pinecone vector store
        
    Returns:
        str: Response to the query
    """
    # Use existing memory from session state
    memory = st.session_state.memory
    
    # Classify query type
    query_type = classify_query(query)
    
    # Route to appropriate handler
    if query_type == "product":
        result = handle_product_query(query, faiss_store, pinecone_store, memory)
    else:
        result = handle_general_query(query, memory)
    
    # Save conversation context
    memory.save_context({"input": query}, {"output": result}) #type: ignore
    
    return result

def handle_product_query(query, faiss_store, pinecone_store, memory):
    """
    Handles product-specific queries using vector stores.
    
    Args:
        query (str): User's query
        faiss_store: FAISS vector store
        pinecone_store: Pinecone vector store
        memory: Conversation memory
        
    Returns:
        str: Product recommendations and information
    """
    # Get appropriate retrievers
    retrievers = select_retrievers(query, faiss_store, pinecone_store)
    
    # Get conversation history
    memory_variables = memory.load_memory_variables({})
    conversation_history = memory_variables.get("history", "")
    
    # Use ensemble retriever for multiple sources
    if len(retrievers) > 1:
        ensemble_retriever = EnsembleRetriever(
            retrievers=retrievers,
            weights=[1/len(retrievers)] * len(retrievers)
        )
        try:
            docs = ensemble_retriever.get_relevant_documents(query)
        except Exception as e:
            print(f"Error with ensemble retriever: {e}")
            docs = []
            for retriever in retrievers:
                try:
                    docs.extend(retriever.get_relevant_documents(query))
                except Exception as e:
                    print(f"Error with individual retriever: {e}")
    elif len(retrievers) == 1:
        try:
            docs = retrievers[0].get_relevant_documents(query)
        except Exception as e:
            print(f"Error with retriever: {e}")
            docs = []
    else:
        return "I couldn't find any specific products matching your criteria. Could you try rephrasing your query or providing more details about what you're looking for?"
    
    if not docs:
        return "I couldn't find any specific products matching your criteria. Could you try rephrasing your query or providing more details about what you're looking for?"
    
    # Format retrieved documents
    formatted_docs = ""
    for i, doc in enumerate(docs[:8]):
        title = doc.metadata.get('Title', 'Unknown Product')
        price = doc.metadata.get('Price', 'Unknown Price')
        category = doc.metadata.get('Category', 'Unknown Category')
        
        formatted_docs += f"PRODUCT {i+1}:\n"
        formatted_docs += f"- Title: {title}\n"
        formatted_docs += f"- Price: ₹{price}\n"
        formatted_docs += f"- Category: {category}\n"
        formatted_docs += f"- Description: {doc.page_content[:200]}...\n\n"
    
    # Initialize language model
    llm = ChatOpenAI(model=llm_model, api_key=openai_api_key) #type: ignore
    
    # Create product information prompt
    template = """
    You are a helpful shopping assistant for health and beauty products.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    CONVERSATION HISTORY:
    {conversation_history}
    
    PRODUCT INFORMATION:
    {formatted_docs}
    
    When answering questions about products:
    1. Use ONLY the product information provided above
    2. Format your response as a numbered list with product name, price, and brief description
    3. Include the exact price for each product as shown in the PRODUCT INFORMATION section
    4. Focus on providing accurate information from the context provided
    5. Consider the conversation history for context, but prioritize the current query
    
    Question: {question}
    
    Answer:"""
    
    PRODUCT_PROMPT = PromptTemplate(
        template=template,
        input_variables=["formatted_docs", "question", "conversation_history"]
    )
    
    # Create and run product information chain
    product_chain = PRODUCT_PROMPT | llm
    result = product_chain.invoke({
        "formatted_docs": formatted_docs, 
        "question": query,
        "conversation_history": conversation_history
    }).content
    
    return result

def handle_general_query(query, memory):
    """
    Handles general information queries using web search or LLM knowledge.
    
    Args:
        query (str): User's query
        memory: Conversation memory
        
    Returns:
        str: General information or advice
    """
    # Get conversation history
    memory_variables = memory.load_memory_variables({})
    conversation_history = memory_variables.get("history", "")
    
    # Handle case when Brave Search API is not available
    if not brave_api_key:
        print("Warning: BRAVE_API_KEY not found in environment variables. Using LLM's knowledge instead.")
        llm = ChatOpenAI(model=llm_model, api_key=openai_api_key) #type: ignore
        
        template = """
        You are a helpful health and beauty advisor.
        The user is asking for general information rather than specific product recommendations.
        
        CONVERSATION HISTORY:
        {conversation_history}
        
        Question: {question}
        
        Answer the question in a helpful, informative way. If providing health advice, remind the user to consult with healthcare professionals for personalized recommendations.
        Consider the conversation history for context, but prioritize the current query.
        """
        
        GENERAL_PROMPT = PromptTemplate(
            template=template,
            input_variables=["question", "conversation_history"]
        )
        
        general_chain = GENERAL_PROMPT | llm
        result = general_chain.invoke({
            "question": query,
            "conversation_history": conversation_history
        }).content
        
        return result
    
    # Use Brave Search for web results
    brave_search = BraveSearch.from_api_key(
        api_key=brave_api_key,
        search_kwargs={"count": 5}
    )
    
    search_results = brave_search.run(query)
    
    # Initialize language model
    llm = ChatOpenAI(model=llm_model, api_key=openai_api_key) #type: ignore
    
    # Create general information prompt
    template = """
    You are a helpful health and beauty advisor.
    Use the following search results to answer the user's question.
    If the search results don't contain relevant information, acknowledge that and provide general advice based on your knowledge.
    
    CONVERSATION HISTORY:
    {conversation_history}
    
    SEARCH RESULTS:
    {search_results}
    
    Question: {question}
    
    Answer the question in a helpful, informative way. If providing health advice, remind the user to consult with healthcare professionals for personalized recommendations.
    Consider the conversation history for context, but prioritize the current query.
    """
    
    # Create a prompt template
    GENERAL_PROMPT = PromptTemplate(
        template=template,
        input_variables=["search_results", "question", "conversation_history"]
    )
    
    # Create a chain for general information
    general_chain = GENERAL_PROMPT | llm
    
    # Run the query with the search results
    result = general_chain.invoke({
        "search_results": search_results, 
        "question": query,
        "conversation_history": conversation_history
    }).content
    
    return result

def create_streamlit_interface():
    """
    Creates a Streamlit interface for the chatbot.
    This function handles:
    - Setting up the page layout and configuration
    - Initializing vector stores for knowledge retrieval
    - Managing chat history and memory
    - Processing user input and displaying responses
    """
    # Set page title and configuration using Streamlit's page settings
    st.set_page_config(
        page_title="Health & Beauty Assistant", # Title shown in browser tab
        layout="wide" # Use wide layout for better readability
    )
    
    # Add header and description text to the page
    st.title("Health & Beauty Assistant") # Main title at top of page
    st.markdown("""
    Ask me about skincare, hair products, vitamins, or general health and beauty advice!
    I can recommend products or provide information based on your needs.
    """) # Descriptive text below title
    
    # Initialize vector stores for knowledge retrieval
    # Uses cached function to avoid reloading on each interaction
    with st.spinner("Loading knowledge base..."): # Show loading spinner
        faiss_store, pinecone_store = initialize_vector_stores("openai", "faiss_index")
    
    # Get existing conversation memory from Streamlit's session state
    memory = st.session_state.memory # Maintains context between interactions
    
    # Initialize empty chat history if first time loading
    if "messages" not in st.session_state:
        st.session_state.messages = [] # List to store conversation history
    
    # Display all previous messages in chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): # Creates chat bubble UI
            st.markdown(message["content"]) # Displays message content
    
    # Create and handle chat input field
    if prompt := st.chat_input("What would you like to know?"): # Get user input
        # Store user's message in chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user's message in chat interface
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process and display assistant's response
        with st.chat_message("assistant"):
            message_placeholder = st.empty() # Create empty placeholder
            message_placeholder.markdown("Thinking...") # Show temporary thinking message
            
            # Get response from query processing function
            response = process_query(prompt, faiss_store, pinecone_store)
            
            # Update placeholder with actual response
            message_placeholder.markdown(response)
        
        # Store assistant's response in chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Main entry point of the application
if __name__ == "__main__":
    create_streamlit_interface() # Start the Streamlit interface
