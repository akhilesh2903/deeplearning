"""
RAG (Retrieval-Augmented Generation) Module
=============================================
Retrieves relevant medical context from knowledge base using:
1. CNN prediction (semantic search)
2. Image embedding similarity
3. Uses FAISS for efficient vector similarity search
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os


class MedicalKnowledgeRetriever:
    """
    Retrieves relevant medical information from knowledge base
    based on CNN predictions and image embeddings.
    """
    
    def __init__(self, knowledge_base_path=None):
        """
        Initialize the retriever with knowledge base and embedding model.
        
        Args:
            knowledge_base_path (str): Path to knowledge_base.json
        """
        # Load knowledge base
        if knowledge_base_path is None:
            knowledge_base_path = os.path.join(
                os.path.dirname(__file__), 'knowledge_base.json'
            )
        
        with open(knowledge_base_path, 'r') as f:
            self.knowledge_base = json.load(f)
        
        # Initialize sentence transformer for embedding similarity
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create embeddings index from knowledge base
        self._build_embedding_index()
    
    def _build_embedding_index(self):
        """Build FAISS index from knowledge base conditions."""
        # Extract all condition texts
        all_texts = []
        self.text_to_condition = {}  # Map text index to condition data
        condition_counter = 0
        
        # Add fetal conditions
        for condition in self.knowledge_base['fetal_conditions']:
            text = f"{condition['condition']}. {condition['description']}"
            all_texts.append(text)
            self.text_to_condition[condition_counter] = {
                'type': 'fetal_condition',
                'data': condition
            }
            condition_counter += 1
        
        # Add maternal conditions
        for condition in self.knowledge_base['maternal_conditions']:
            text = f"{condition['condition']}. {condition['effect_on_fetus']}"
            all_texts.append(text)
            self.text_to_condition[condition_counter] = {
                'type': 'maternal_condition',
                'data': condition
            }
            condition_counter += 1
        
        # Create embeddings for all texts
        embeddings = self.embedding_model.encode(all_texts)
        embeddings = np.array(embeddings).astype('float32')
        
        # Build FAISS index
        self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.faiss_index.add(embeddings)
    
    def retrieve_by_condition(self, predicted_class_label, top_k=3):
        """
        Retrieve relevant medical context based on CNN prediction.
        
        Args:
            predicted_class_label (str): Predicted condition from CNN
            top_k (int): Number of top results to retrieve
            
        Returns:
            dict: Retrieved medical contexts
        """
        # Create embedding for prediction
        query_embedding = self.embedding_model.encode([predicted_class_label])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search FAISS index
        distances, indices = self.faiss_index.search(query_embedding, top_k + 5)
        
        # Retrieve matched conditions
        matched_conditions = []
        for idx in indices[0]:
            if idx in self.text_to_condition:
                condition_info = self.text_to_condition[idx]
                matched_conditions.append(condition_info)
        
        # Filter to return top_k most relevant
        result = {
            'predicted_condition': predicted_class_label,
            'related_fetal_conditions': [],
            'related_maternal_conditions': [],
            'doppler_parameters': self.knowledge_base.get('doppler_parameters', []),
            'biometric_measurements': self.knowledge_base.get('biometric_measurements', [])
        }
        
        for condition_info in matched_conditions[:top_k]:
            if condition_info['type'] == 'fetal_condition':
                result['related_fetal_conditions'].append(condition_info['data'])
            else:
                result['related_maternal_conditions'].append(condition_info['data'])
        
        return result
    
    def retrieve_by_embedding(self, image_embedding, top_k=2):
        """
        Retrieve medical context based on image embedding similarity.
        (Optional: Use if you have pre-computed embeddings for conditions)
        
        Args:
            image_embedding (np.ndarray): Image feature embedding from CNN
            top_k (int): Number of top results
            
        Returns:
            dict: Similar conditions from knowledge base
        """
        # Normalize embedding
        image_embedding = np.array([image_embedding]).astype('float32')
        
        # Search similar embeddings
        distances, indices = self.faiss_index.search(image_embedding, top_k)
        
        similar_conditions = []
        for idx in indices[0]:
            if idx in self.text_to_condition:
                similar_conditions.append(self.text_to_condition[idx])
        
        return {
            'similar_conditions': similar_conditions,
            'distances': distances[0].tolist()
        }
    
    def get_all_maternal_risk_factors(self):
        """Return all maternal risk factors with their fetal impact."""
        return self.knowledge_base['maternal_conditions']
    
    def get_doppler_guidelines(self):
        """Return Doppler measurement interpretation guidelines."""
        return self.knowledge_base['doppler_parameters']
    
    def get_risk_stratification(self):
        """Return risk stratification guidelines."""
        return self.knowledge_base['risk_stratification']
    
    def retrieve_combined_context(self, predicted_label, confidence, image_embedding=None, top_k=3):
        """
        Retrieve comprehensive medical context for report generation.
        
        Args:
            predicted_label (str): Predicted condition from CNN
            confidence (float): Confidence score from CNN
            image_embedding (np.ndarray): Optional image embedding
            top_k (int): Number of top conditions to retrieve
            
        Returns:
            dict: Comprehensive medical context for LLM
        """
        # Get condition-based retrieval
        condition_context = self.retrieve_by_condition(predicted_label, top_k)
        
        # Add risk stratification based on confidence
        risk_strat = self.get_risk_stratification()
        
        # Determine risk level based on confidence
        if confidence > 80:
            risk_level = [r for r in risk_strat if 'High' in r['risk_level']]
        elif confidence > 60:
            risk_level = [r for r in risk_strat if 'Moderate' in r['risk_level']]
        else:
            risk_level = [r for r in risk_strat if 'Low' in r['risk_level']]
        
        result = {
            'predicted_condition': predicted_label,
            'confidence_level': confidence,
            'retrieved_conditions': condition_context,
            'applicable_risk_stratification': risk_level[0] if risk_level else risk_strat[0],
            'doppler_guidelines': self.get_doppler_guidelines(),
            'maternal_risk_factors': self.get_all_maternal_risk_factors()[:3]  # Top 3
        }
        
        return result
