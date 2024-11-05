# RAG (Retrieval Augmented Generation) Application for EDU-TECH

## Overview
This application is a sophisticated document query system that implements RAG (Retrieval Augmented Generation) architecture. It allows users to upload PDF documents and ask questions about their content, receiving AI-generated responses based on the document's content.

## Technologies Used

### Core Technologies
- **Streamlit**: Front-end framework for creating the web interface
- **LangChain**: Framework for building LLM applications
- **Groq**: LLM provider using the llama-3.1-70b-versatile model
- **Google Generative AI**: Used for generating embeddings
- **ChromaDB**: Vector store for document embeddings
- **Python**: Primary programming language

### Key Libraries
- `langchain_community.document_loaders.PyPDFLoader`: For loading and processing PDF files
- `langchain.text_splitter.RecursiveCharacterTextSplitter`: For splitting documents into manageable chunks
- `langchain_community.retrievers.BM25Retriever`: For keyword-based retrieval
- `langchain_community.vectorstores.Chroma`: For vector storage and retrieval
- `python-dotenv`: For environment variable management

## Features

### 1. Document Processing
- PDF document upload capability
- Automatic document chunking with configurable size and overlap
- Text extraction and processing
- Temporary file handling for uploaded documents

### 2. Advanced Retrieval System
- **Hybrid Retrieval**: Combines two retrieval methods:
  - Vector-based similarity search (70% weight)
  - BM25 keyword-based search (30% weight)
- Ensemble retriever for improved accuracy
- Persistent vector storage

### 3. Natural Language Processing
- Integration with Groq's LLM for response generation
- Custom prompt template with step-by-step reasoning
- Google's Generative AI embeddings for document vectorization

### 4. User Interface
- Clean, intuitive web interface
- Real-time document processing
- Interactive query system
- Progress indicators and success/error messages

## System Architecture

### Document Processing Flow
1. User uploads PDF document
2. Document is temporarily stored and processed
3. Text is extracted and split into chunks
4. Chunks are converted to embeddings
5. Embeddings are stored in ChromaDB

### Query Processing Flow
1. User submits question
2. Question is processed by the ensemble retriever
3. Relevant document chunks are retrieved
4. LLM generates response based on retrieved content
5. Response is displayed to user

## State Management
- Uses Streamlit's session state for managing:
  - Processed file tracking
  - Vector store persistence
  - Document chunks
  - Retrieval chain configuration

## Error Handling
- Comprehensive error handling for:
  - Document loading failures
  - Processing errors
  - Query execution issues
  - API communication problems

## Environment Configuration
Required environment variables:
- `GOOGLE_API_KEY`: For Google Generative AI services
- `GROQ_API_KEY`: For accessing Groq's LLM services

## Performance Features
- Caching of processed documents
- Efficient document chunking
- Optimized retrieval through ensemble methods
- Persistent storage for vector embeddings

## Technical Details

### Document Chunking
- Chunk size: 1000 characters
- Chunk overlap: 200 characters
- Uses recursive character splitting for optimal chunk creation

### Retrieval Configuration
- Vector similarity search weight: 0.7
- BM25 keyword search weight: 0.3
- Customized prompt template for improved response generation

## Usage Limitations
- Supports PDF files only
- Requires active internet connection
- API keys must be properly configured
- Memory usage scales with document size

## Security Considerations
- Temporary file handling with automatic cleanup
- Environment variable protection
- No permanent storage of uploaded files

This application represents a modern approach to document question-answering, combining multiple retrieval methods with state-of-the-art language models to provide accurate and contextual responses to user queries.
## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

