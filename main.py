import os
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import EnsembleRetriever
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
import re
import pinecone
from langchain_community.tools.brave_search.tool import BraveSearch
import streamlit as st

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
brave_api_key = os.getenv("BRAVE_API_KEY")

# Initialize embeddings outside of functions to avoid PyTorch conflicts
@st.cache_resource
def get_embeddings(model: str):
    if model == "openai":
        return OpenAIEmbeddings(model = "text-embedding-3-small", api_key = "") #type: ignore
    else:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Use st.cache_resource to ensure this function runs only once
@st.cache_resource
def initialize_vector_stores(model:str):
    """
    Initialize and return vector stores (FAISS and Pinecone).
    If Pinecone index already exists, skip document ingestion.
    
    Returns:
        tuple: (faiss_store, pinecone_store) - The initialized vector stores
    """
    # Get embeddings
    embeddings = get_embeddings(model)
    if model == "openai":
        embedding_size = 1536
    else:
        embedding_size = 384
    
    file_paths = ["df_skin.csv", "df_hair.csv", "df_vits_supp.csv"]
    namespaces = {"df_skin.csv": "skin", "df_hair.csv": "hair", "df_vits_supp.csv": "vitamins_supplements"}
    faiss_docs = []
    pinecone_docs = {"skin": [], "hair": [], "vitamins_supplements": []}

    # Initialize Pinecone with the updated API
    pc = pinecone.Pinecone(api_key=pinecone_api_key)

    # Check if index exists
    if model == "openai":
        index_name = "clinikally-rag-2"
    else:
        index_name = "clinikally-rag"
    index_exists = False

    try:
        # List all indexes and check if our index exists
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

    # Process documents for FAISS regardless of Pinecone status
    for file in file_paths:
        # Load the CSV file
        loader = CSVLoader(file_path=file)
        documents = loader.load()
        
        # Process documents to extract title and price before splitting
        for doc in documents:
            # Extract title and price from the content
            content_lines = doc.page_content.split('\n')
            title = ""
            price = ""
            
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
            
            # Add these to metadata
            doc.metadata["Title"] = title
            doc.metadata["Price"] = price
            doc.metadata["Brand"] = brand

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = text_splitter.split_documents(documents)
        namespace = namespaces[file]
        
        for doc in split_docs:
            doc.metadata["Category"] = namespace
            # Title and Price are already in metadata from the parent document
            faiss_docs.append(doc)
            
            # Only add to pinecone_docs if we need to ingest
            if not index_exists:
                pinecone_docs[namespace].append(doc)
        
        print(f"Number of documents in {namespace}: {len(split_docs)}")
        print(f"Number of documents in Faiss: {len(faiss_docs)}")

    # Create FAISS store
    faiss_store = FAISS.from_documents(faiss_docs, embeddings)

    # Create Pinecone vector stores
    pinecone_store = {}
    for namespace in namespaces.values():
        # Connect to existing vector store without adding documents
        vector_store = PineconeVectorStore(
            index_name=index_name, 
            embedding=embeddings, 
            text_key="text", 
            namespace=namespace
        )
        
        # Only add documents if the index didn't exist before
        if not index_exists and namespace in pinecone_docs and pinecone_docs[namespace]:
            print(f"Ingesting documents to Pinecone namespace: {namespace}")
            vector_store.add_documents(pinecone_docs[namespace])
        
        pinecone_store[namespace] = vector_store
    
    return faiss_store, pinecone_store

def classify_query(query):
    """
    Classifies a query as either product-based or general information.
    
    Args:
        query (str): The user's query
        
    Returns:
        str: Either "product" or "general"
    """
    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=openai_api_key) #type: ignore
    
    # Create a prompt template for classification
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
    
    # Create a chain for classification
    from langchain.chains import LLMChain
    classification_chain = LLMChain(llm=llm, prompt=classification_prompt)
    
    # Run the classification
    result = classification_chain.run(query).strip().lower()
    
    # Ensure we get a valid result
    if result not in ["product", "general"]:
        # If the model didn't follow instructions, use heuristics as fallback
        product_keywords = ["product", "buy", "purchase", "recommend", "price", "cost", "rupees", "rs", "₹", "brand", "where to get", "suggest", "suggestions"]
        general_keywords = ["how to", "why", "what causes", "explain", "treatment", "remedy", "cure", "prevent", "tips", "advice"]
        
        product_score = sum(1 for keyword in product_keywords if keyword in query.lower())
        general_score = sum(1 for keyword in general_keywords if keyword in query.lower())
        
        result = "product" if product_score >= general_score else "general"
    
    print(f"Query classified as: {result}")
    return result

def select_retrievers(query, faiss_store, pinecone_store):
    selected_retrievers = []
    query_lower = query.lower()
    
    # Initialize filter dictionary
    filter_dict = {}
    
    # Extract price filter from query if present
    # Check for "under X rupees" pattern
    under_match = re.search(r'under\s+(\d+(?:\.\d+)?)\s+(?:rs\.?|rupees?)', query_lower)
    if under_match:
        try:
            price_threshold = float(under_match.group(1))
            filter_dict["Price"] = {"$lt": price_threshold}
        except ValueError:
            pass
    
    # Check for "less than X rupees" pattern
    less_than_match = re.search(r'less\s+than\s+(\d+(?:\.\d+)?)\s+(?:rs\.?|rupees?)', query_lower)
    if less_than_match and "Price" not in filter_dict:
        try:
            price_threshold = float(less_than_match.group(1))
            filter_dict["Price"] = {"$lt": price_threshold}
        except ValueError:
            pass
    
    # Check for "between X and Y rupees" pattern
    between_match = re.search(r'between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s+(?:rs\.?|rupees?)', query_lower)
    if between_match:
        try:
            min_price = float(between_match.group(1))
            max_price = float(between_match.group(2))
            filter_dict["Price"] = {"$gte": min_price, "$lte": max_price}
        except ValueError:
            pass
    
    # Extract brand filter
    # Common brands in the dataset
    brands = [
        "Abbott", "Melaglow", "Episoft", "Bioderma", "Neutriderm", "Follihair", 
        "Senechio Pharma", "Glenmark", "Epique", "Blue Cap", "Acnechio", "GM"
    ]
    
    # Check for brand mentions - using $eq instead of $contains for Pinecone compatibility
    for brand in brands:
        if brand.lower() in query_lower:
            filter_dict["Brand"] = {"$eq": brand}
            break
    
    # Extract rating filter
    # Check for "rating above X" or "rated above X" pattern
    rating_above_match = re.search(r'rat(?:ing|ed)\s+(?:above|over)\s+(\d+(?:\.\d+)?)', query_lower)
    if rating_above_match:
        try:
            rating_threshold = float(rating_above_match.group(1))
            filter_dict["reviews.rating_count"] = {"$gt": rating_threshold}
        except ValueError:
            pass
    
    # Check for "X star" or "X stars" pattern
    stars_match = re.search(r'(\d+(?:\.\d+)?)\s+stars?', query_lower)
    if stars_match and "reviews.rating_count" not in filter_dict:
        try:
            rating_threshold = float(stars_match.group(1))
            filter_dict["reviews.rating_count"] = {"$gte": rating_threshold}
        except ValueError:
            pass
    
    # Define regex patterns for each category
    patterns = {
        "skin": r'\b(skin|face|acne|pimple|complexion|blemish|wrinkle|dark spot|dark circle|pore|blackhead|whitehead|rash|dermatitis|eczema|psoriasis|rosacea|pigmentation|scar|aging|moisturizer|cleanser|toner|serum|sunscreen|skincare)\b',
        "hair": r'\b(hair|scalp|dandruff|hairfall|hair loss|hair growth|split end|frizz|dry hair|oily hair|hair care|shampoo|conditioner|hair mask|hair treatment|hair color|hair dye|hair style|hair product|balding|thinning|alopecia|grey hair|hair texture|hair volume)\b',
        "vitamins_supplements": r'\b(vitamin|supplement|mineral|nutrition|nutrient|deficiency|dietary|multivitamin|antioxidant|omega|protein|calcium|iron|magnesium|zinc|potassium|biotin|collagen|probiotic|prebiotic|amino acid|herbal|natural supplement|wellness|immunity|energy|metabolism)\b'
    }
    
    # Check each category's pattern against the query
    matched_categories = []
    for category, pattern in patterns.items():
        if re.search(pattern, query_lower):
            matched_categories.append(category)
    
    # If no categories matched, use all of them
    if not matched_categories:
        matched_categories = list(pinecone_store.keys())
    
    # Add retrievers for matched categories
    for category in matched_categories:
        if category in pinecone_store:
            try:
                if filter_dict:
                    selected_retrievers.append(pinecone_store[category].as_retriever(search_kwargs={"filter": filter_dict}))
                else:
                    selected_retrievers.append(pinecone_store[category].as_retriever())
            except Exception as e:
                print(f"Error creating retriever for {category}: {e}")
                # Fall back to retriever without filter
                selected_retrievers.append(pinecone_store[category].as_retriever())
    
    # Always add FAISS as a fallback
    try:
        if filter_dict:
            selected_retrievers.append(faiss_store.as_retriever(search_kwargs={"filter": filter_dict}))
        else:
            selected_retrievers.append(faiss_store.as_retriever())
    except Exception as e:
        print(f"Error creating FAISS retriever: {e}")
        # Fall back to retriever without filter
        selected_retrievers.append(faiss_store.as_retriever())
    
    # Print the filters being applied (for debugging)
    if filter_dict:
        print(f"Applying filters: {filter_dict}")
    
    return selected_retrievers

def safe_retrieval(retriever, query):
    """Safely retrieve documents with error handling"""
    try:
        return retriever.invoke(query)
    except Exception as e:
        print(f"Error during retrieval: {e}")
        # Return empty list on error
        return []

# Initialize memory function
@st.cache_resource
def get_memory():
    """
    Initialize and return a ConversationSummaryBufferMemory instance.
    
    Returns:
        ConversationSummaryBufferMemory: The initialized memory
    """
    return ConversationSummaryBufferMemory(
        llm=ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_api_key), #type: ignore
        max_token_limit=250,
        return_messages=True
    )

def process_query(query, faiss_store, pinecone_store):
    """
    Main function to process user queries.
    Determines query type and routes to appropriate handler.
    Uses memory to maintain conversation context.
    
    Args:
        query (str): The user's query
        faiss_store: The FAISS vector store
        pinecone_store: The Pinecone vector store
        
    Returns:
        str: The response to the query
    """
    # Get memory
    memory = get_memory()
    
    # Classify the query
    query_type = classify_query(query)
    
    # Handle the query based on its type
    if query_type == "product":
        result = handle_product_query(query, faiss_store, pinecone_store, memory)
    else:  # general query
        result = handle_general_query(query, memory)
    
    # Save the conversation to memory
    memory.save_context({"input": query}, {"output": result})
    
    return result

def handle_product_query(query, faiss_store, pinecone_store, memory):
    """
    Handles product-based queries by retrieving relevant product information.
    
    Args:
        query (str): The user's query
        faiss_store: The FAISS vector store
        pinecone_store: The Pinecone vector store
        memory: The conversation memory
        
    Returns:
        str: The response to the query
    """
    # Get retrievers based on the query
    retrievers = select_retrievers(query, faiss_store, pinecone_store)
    
    # Get conversation history
    memory_variables = memory.load_memory_variables({})
    conversation_history = memory_variables.get("history", "")
    
    # Create an ensemble retriever if we have multiple retrievers
    if len(retrievers) > 1:
        ensemble_retriever = EnsembleRetriever(
            retrievers=retrievers,
            weights=[1/len(retrievers)] * len(retrievers)
        )
        try:
            docs = ensemble_retriever.get_relevant_documents(query)
        except Exception as e:
            print(f"Error with ensemble retriever: {e}")
            # Try individual retrievers if ensemble fails
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
    
    # Format the documents with their metadata explicitly included in the text
    formatted_docs = ""
    for i, doc in enumerate(docs[:8]):  # Limit to first 8 for response
        title = doc.metadata.get('Title', 'Unknown Product')
        price = doc.metadata.get('Price', 'Unknown Price')
        category = doc.metadata.get('Category', 'Unknown Category')
        
        formatted_docs += f"PRODUCT {i+1}:\n"
        formatted_docs += f"- Title: {title}\n"
        formatted_docs += f"- Price: ₹{price}\n"
        formatted_docs += f"- Category: {category}\n"
        formatted_docs += f"- Description: {doc.page_content[:200]}...\n\n"
    
    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=openai_api_key) #type: ignore
    
    # Create a prompt template for product information
    template = """
    You are a helpful shopping assistant for health and beauty products.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    PRODUCT INFORMATION:
    {formatted_docs}
    
    When answering questions about products:
    1. Use ONLY the product information provided above
    2. Format your response as a numbered list with product name, price, and brief description
    3. Include the exact price for each product as shown in the PRODUCT INFORMATION section
    4. Focus on providing accurate information from the context provided
    
    Question: {question}
    
    Answer:"""
    
    # Create a prompt template with the formatted documents
    PRODUCT_PROMPT = PromptTemplate(
        template=template,
        input_variables=["formatted_docs", "question"]
    )
    
    # Create a chain for product information
    from langchain.chains import LLMChain
    product_chain = LLMChain(llm=llm, prompt=PRODUCT_PROMPT)
    
    # Run the query with the formatted documents
    result = product_chain.run(formatted_docs=formatted_docs, question=query)
    
    return result

def handle_general_query(query, memory):
    """
    Handles general information queries using BraveSearch.
    
    Args:
        query (str): The user's query
        memory: The conversation memory
        
    Returns:
        str: The response to the query
    """
    # Get conversation history
    memory_variables = memory.load_memory_variables({})
    conversation_history = memory_variables.get("history", "")
    
    # Check if brave_api_key is available
    if not brave_api_key:
        print("Warning: BRAVE_API_KEY not found in environment variables. Using LLM's knowledge instead.")
        # Initialize the language model
        llm = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=openai_api_key) #type: ignore
        
        # Create a prompt template for general information without search results
        template = """
        You are a helpful health and beauty advisor.
        The user is asking for general information rather than specific product recommendations.
        
        Question: {question}
        
        Answer the question in a helpful, informative way. If providing health advice, remind the user to consult with healthcare professionals for personalized recommendations.
        """
        
        # Create a prompt template
        GENERAL_PROMPT = PromptTemplate(
            template=template,
            input_variables=["question"]
        )
        
        # Create a chain for general information
        from langchain.chains import LLMChain
        general_chain = LLMChain(llm=llm, prompt=GENERAL_PROMPT)
        
        # Run the query without search results
        result = general_chain.run(question=query)
        
        return result
    
    # Initialize BraveSearch tool
    brave_search = BraveSearch.from_api_key(
        api_key=brave_api_key,
        search_kwargs={"count": 5}  # Limit to 5 results for relevance
    )
    
    # Get search results
    search_results = brave_search.run(query)
    
    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=openai_api_key) #type: ignore
    
    # Create a prompt template for general information
    template = """
    You are a helpful health and beauty advisor.
    Use the following search results to answer the user's question.
    If the search results don't contain relevant information, acknowledge that and provide general advice based on your knowledge.
    
    SEARCH RESULTS:
    {search_results}
    
    Question: {question}
    
    Answer the question in a helpful, informative way. If providing health advice, remind the user to consult with healthcare professionals for personalized recommendations.
    """
    
    # Create a prompt template
    GENERAL_PROMPT = PromptTemplate(
        template=template,
        input_variables=["search_results", "question"]
    )
    
    # Create a chain for general information
    from langchain.chains import LLMChain
    general_chain = LLMChain(llm=llm, prompt=GENERAL_PROMPT)
    
    # Run the query with the search results
    result = general_chain.run(search_results=search_results, question=query)
    
    return result

def create_streamlit_interface():
    """
    Creates a Streamlit interface for the chatbot.
    """
    # Set page title and configuration
    st.set_page_config(
        page_title="Health & Beauty Assistant",
        page_icon="💄",
        layout="wide"
    )
    
    # Add header and description
    st.title("Health & Beauty Assistant")
    st.markdown("""
    Ask me about skincare, hair products, vitamins, or general health and beauty advice!
    I can recommend products or provide information based on your needs.
    """)
    
    # Add model selection dropdown
    model = st.selectbox(
        "Choose the model",
        ["openai", "generic"],
        help="Select which model to use for embeddings and retrieval"
    )
    
    # Initialize vector stores - this will only run once due to @st.cache_resource
    with st.spinner("Loading knowledge base..."):
        faiss_store, pinecone_store = initialize_vector_stores(model)
    
    # Initialize memory - this will only run once due to @st.cache_resource
    memory = get_memory()
    
    # Initialize chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            # Process the query
            response = process_query(prompt, faiss_store, pinecone_store)
            
            # Update the placeholder with the response
            message_placeholder.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Entry point
if __name__ == "__main__":
    create_streamlit_interface()
