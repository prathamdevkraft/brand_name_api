#!/usr/bin/env python3
"""
AI-Enhanced Brand identification service for FastAPI
"""

import re
import os
from typing import Dict, List, Optional, Any
from collections import Counter
import time
import logging
import asyncio

# AI/ML imports
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)

class AIBrandIdentifierService:
    """AI-Enhanced brand identification service"""
    
    def __init__(self, settings):
        self.settings = settings
        self.cache = {}  # Simple in-memory cache
        self._init_ai_services()
        
    def _init_ai_services(self):
        """Initialize AI services"""
        # OpenAI setup
        if HAS_OPENAI and self.settings.openai_api_key:
            openai.api_key = self.settings.openai_api_key
            self.has_openai = True
            logger.info("OpenAI service initialized")
        else:
            self.has_openai = False
            logger.warning("OpenAI not available - missing API key or library")
            
        # spaCy setup
        if HAS_SPACY:
            try:
                #self.nlp = spacy.load(self.settings.spacy_model)
                self.nlp = spacy.load("xx_ent_wiki_sm")
                self.has_spacy = True
                logger.info(f"spaCy model '{self.settings.spacy_model}' loaded")
            except OSError:
                self.has_spacy = False
                logger.warning(f"spaCy model '{self.settings.spacy_model}' not found")
        else:
            self.has_spacy = False
            logger.warning("spaCy not available")
            
        # Transformers setup
        if HAS_TRANSFORMERS:
            try:
                self.ner_pipeline = pipeline(
                    "ner", 
                    #model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    #model="Davlan/xlm-roberta-base-ner-hrl",
                    model="dslim/bert-base-NER",
                    aggregation_strategy="simple"
                )
                self.has_transformers = True
                logger.info("Transformers NER pipeline loaded")
            except Exception as e:
                self.has_transformers = False
                logger.warning(f"Transformers not available: {e}")
        else:
            self.has_transformers = False
    
    async def identify_brand(self, file_path: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        AI-Enhanced brand identification from HTML file
        
        Args:
            file_path: Path to HTML file
            filename: Original filename for context
            
        Returns:
            Dictionary with brand identification results
        """
        # Check cache first
        file_key = self._get_file_key(file_path)
        if file_key in self.cache:
            cached_result = self.cache[file_key].copy()
            cached_result['method'] = f"{cached_result['method']}_cached"
            return cached_result
        
        # Load HTML content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
        # Run all extraction methods concurrently
        results = []
        
        # Traditional methods (fast)
        if filename:
            filename_result = self._extract_from_filename(filename)
            if filename_result:
                results.append(filename_result)
        
        pattern_result = self._extract_patterns(html_content)
        if pattern_result:
            results.append(pattern_result)
            
        structure_result = self._analyze_structure(html_content)
        if structure_result:
            results.append(structure_result)
        
        # AI methods (slower but more accurate)
        ai_tasks = []
        
        print(f"🔍 AI Services Status: OpenAI={self.has_openai}, spaCy={self.has_spacy}, Transformers={self.has_transformers}")
        
        if self.has_openai:
            print("🤖 Adding OpenAI extraction task...")
            ai_tasks.append(self._openai_extraction(html_content))
            
        if self.has_spacy:
            print("🧠 Adding spaCy extraction task...")
            ai_tasks.append(self._spacy_extraction(html_content))
            
        if self.has_transformers:
            print("🔬 Adding Transformers extraction task...")
            ai_tasks.append(self._transformers_extraction(html_content))
        
        # Run AI methods concurrently
        if ai_tasks:
            try:
                print(f"🚀 Running {len(ai_tasks)} AI methods...")
                ai_results = await asyncio.gather(*ai_tasks, return_exceptions=True)
                for i, result in enumerate(ai_results):
                    if isinstance(result, dict) and result:
                        print(f"✅ AI method {i+1} succeeded: {result}")
                        results.append(result)
                    elif isinstance(result, Exception):
                        print(f"❌ AI method {i+1} failed: {result}")
                        logger.error(f"AI extraction failed: {result}")
                    else:
                        print(f"⚠️ AI method {i+1} returned: {result}")
            except Exception as e:
                print(f"💥 Error running AI methods: {e}")
                logger.error(f"Error running AI methods: {e}")
        
        # Combine all results with AI-enhanced ensemble
        final_result = self._ai_ensemble_voting(results)
        
        # Cache result
        self.cache[file_key] = final_result
        
        return final_result
    
    def _get_file_key(self, file_path: str) -> str:
        """Generate cache key for file"""
        try:
            stat = os.stat(file_path)
            return f"{file_path}_{stat.st_size}_{stat.st_mtime}"
        except:
            return file_path
    
    def _extract_from_filename(self, filename: str) -> Optional[Dict]:
        """Extract brand from filename"""
        name = os.path.splitext(filename)[0]
        brand_match = re.match(r'^([A-Za-z]+)', name)
        
        if brand_match:
            brand = brand_match.group(1)
            if len(brand) >= 3 and brand.lower() not in ['index', 'main', 'home', 'template', 'file']:
                return {
                    'brand': brand.title(),
                    'confidence': 0.9,
                    'method': 'filename',
                    'source': filename
                }
        return None
    
    def _extract_patterns(self, html_content: str) -> Optional[Dict]:
        """Extract brand using universal patterns"""
        patterns = [
            (r'src="[^"]*([A-Za-z]{3,})[_-]?[Ll]ogo\.', 0.85),
            (r'([A-Z][A-Z0-9]{2,})[®™]', 0.9),
            (r'https?://(?:www\.)?([a-z]{3,})(?:hcp)?\.com', 0.8),
            (r'class="[^"]*([A-Z][a-z]{2,})[_-]', 0.7),
            (r'alt="[^"]*([A-Z][a-z]{2,})[^"]*[Ll]ogo', 0.8),
            (r'([A-Z][A-Z0-9]{2,})[_-][Ll]ogo', 0.85)
        ]
        
        candidates = []
        for pattern, confidence in patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                if len(match) >= 3 and match.isalpha():
                    candidates.append((match.title(), confidence))
        
        if candidates:
            brand_counts = Counter([brand for brand, _ in candidates])
            if brand_counts:
                most_common_brand = brand_counts.most_common(1)[0][0]
                confidences = [conf for brand, conf in candidates if brand == most_common_brand]
                avg_confidence = sum(confidences) / len(confidences)
                
                return {
                    'brand': most_common_brand,
                    'confidence': avg_confidence,
                    'method': 'patterns',
                    'matches': len(confidences)
                }
        return None
    
    def _analyze_structure(self, html_content: str) -> Optional[Dict]:
        """Analyze HTML structure for brand indicators"""
        indicators = []
        
        # Title, meta, headings, trademark analysis
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
        if title_match:
            title = title_match.group(1)
            brand_match = re.search(r'\b([A-Z][a-z]{2,})\b', title)
            if brand_match and brand_match.group(1).lower() not in ['email', 'template', 'html']:
                indicators.append((brand_match.group(1), 0.7))
        
        # Strong/bold with trademarks
        strong_matches = re.findall(r'<(?:strong|b)[^>]*>([^<]*[®™][^<]*)</(?:strong|b)>', html_content, re.IGNORECASE)
        for strong_text in strong_matches:
            brand_match = re.search(r'\b([A-Z][a-z]{2,})', strong_text)
            if brand_match:
                indicators.append((brand_match.group(1), 0.9))
        
        if indicators:
            brand_counts = Counter([brand for brand, _ in indicators])
            if brand_counts:
                most_common_brand = brand_counts.most_common(1)[0][0]
                confidences = [conf for brand, conf in indicators if brand == most_common_brand]
                avg_confidence = sum(confidences) / len(confidences)
                
                return {
                    'brand': most_common_brand,
                    'confidence': avg_confidence,
                    'method': 'structure',
                    'indicators': len(confidences)
                }
        return None
    
    async def _openai_extraction(self, html_content: str) -> Optional[Dict]:
        """Use OpenAI for intelligent brand extraction"""
        if not self.has_openai:
            return None
            
        try:
            # Prepare content (truncate for efficiency)
            text_content = re.sub(r'<[^>]+>', ' ', html_content)[:2000]
            
            prompt = f"""
You are a pharmaceutical expert specializing in drug brand identification.

TASK: Extract the main PHARMACEUTICAL BRAND NAME (drug product name) from this HTML content.

WHAT TO LOOK FOR:
- Drug brand names like: Biktarvy, Camzyos, Lynparza, Humira, Lipitor, Ozempic, Keytruda
- Names with trademark symbols (®, ™)
- Names in titles, headings, or prominent text
- Names mentioned multiple times
- Names associated with dosages (mg, tablets, injection)

WHAT TO IGNORE:
- Medical conditions: HIV, AIDS, cancer, diabetes, hypertension, depression
- Company names: Pfizer, Gilead, Bristol Myers Squibb (unless they're also the drug name)
- Generic terms: treatment, therapy, medication, drug
- Common words: patient, clinical, study, information

EXAMPLES:
✓ "BIKTARVY® (bictegravir/emtricitabine/tenofovir)" → Return: Biktarvy
✓ "CAMZYOS® for hypertrophic cardiomyopathy" → Return: Camzyos  
✗ "HIV treatment options" → This is a condition, not a brand
✗ "Gilead Sciences" → This is a company, not a drug brand

INSTRUCTIONS: Return ONLY the brand name, nothing else. If no pharmaceutical brand found, return "UNKNOWN".

Content: {text_content}

Pharmaceutical brand name:"""
            
            def make_openai_call():
                client = openai.OpenAI(api_key=self.settings.openai_api_key)
                return client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=20,
                    temperature=0.1
                )
            
            response = await asyncio.get_event_loop().run_in_executor(None, make_openai_call)
            
            brand = response.choices[0].message.content.strip()
            
            if brand and brand != "UNKNOWN" and len(brand) >= 3 and brand.isalpha():
                return {
                    'brand': brand.title(),
                    'confidence': 0.95,
                    'method': 'openai_llm',
                    'model': 'gpt-3.5-turbo'
                }
        
        except Exception as e:
            logger.error(f"OpenAI extraction failed: {e}")
        
        return None
    
    async def _spacy_extraction(self, html_content: str) -> Optional[Dict]:
        """Use spaCy for Named Entity Recognition"""
        if not self.has_spacy:
            return None
            
        try:
            # Extract text and truncate
            text = re.sub(r'<[^>]+>', ' ', html_content)[:1000]
            
            # Run spaCy NER
            doc = await asyncio.get_event_loop().run_in_executor(
                None, self.nlp, text
            )
            
            # Extract entities
            brand_candidates = []
            for ent in doc.ents:
                if (ent.label_ in ['ORG', 'PRODUCT'] and 
                    len(ent.text) >= 3 and 
                    ent.text.isalpha() and
                    ent.text[0].isupper()):
                    brand_candidates.append(ent.text)
            
            if brand_candidates:
                # Get most frequent
                brand_counts = Counter(brand_candidates)
                most_common = brand_counts.most_common(1)[0][0]
                
                return {
                    'brand': most_common.title(),
                    'confidence': 0.8,
                    'method': 'spacy_ner',
                    'entities_found': len(brand_candidates)
                }
        
        except Exception as e:
            logger.error(f"spaCy extraction failed: {e}")
        
        return None
    
    async def _transformers_extraction(self, html_content: str) -> Optional[Dict]:
        """Use Transformers for advanced NER"""
        if not self.has_transformers:
            return None
            
        try:
            # Extract and truncate text
            text = re.sub(r'<[^>]+>', ' ', html_content)[:500]
            
            # Run NER pipeline
            ner_results = await asyncio.get_event_loop().run_in_executor(
                None, self.ner_pipeline, text
            )
            
            # Extract brand candidates with pharmaceutical filtering
            brand_candidates = []
            
            # Filter out medical conditions and common words
            medical_terms = {'hiv', 'aids', 'cancer', 'diabetes', 'heart', 'disease', 'therapy', 'treatment'}
            common_words = {'explore', 'information', 'patient', 'clinical', 'study', 'trial', 'ion'}
            
            for entity in ner_results:
                if (entity['entity_group'] in ['ORG', 'MISC', 'PER'] and
                    entity['score'] > 0.7 and
                    len(entity['word']) >= 3 and
                    entity['word'].replace('#', '').isalpha()):
                    clean_word = entity['word'].replace('#', '').strip()
                    clean_lower = clean_word.lower()
                    
                    # Skip medical terms and common words
                    if (clean_word[0].isupper() and 
                        clean_lower not in medical_terms and
                        clean_lower not in common_words and
                        len(clean_word) >= 4):  # Pharmaceutical brands are usually 4+ chars
                        brand_candidates.append(clean_word)
            
            if brand_candidates:
                brand_counts = Counter(brand_candidates)
                most_common = brand_counts.most_common(1)[0][0]
                
                return {
                    'brand': most_common.title(),
                    'confidence': 0.85,
                    'method': 'transformers_ner',
                    'entities_found': len(brand_candidates)
                }
        
        except Exception as e:
            logger.error(f"Transformers extraction failed: {e}")
        
        return None
    
    def _ai_ensemble_voting(self, results: List[Dict]) -> Dict[str, Any]:
        """AI-Enhanced ensemble voting with method weighting"""
        if not results:
            return {
                'brand': 'UNKNOWN',
                'confidence': 0.0,
                'method': 'none',
                'details': {'message': 'No brand could be identified'}
            }
        
        if len(results) == 1:
            result = results[0]
            return {
                'brand': result['brand'],
                'confidence': result['confidence'],
                'method': result['method'],
                'details': result
            }
        
        # Method weights (AI methods get higher weights)
        method_weights = {
            'filename': 0.9,
            'patterns': 0.8,
            'structure': 0.7,
            'openai_llm': 1.0,      # Highest weight for LLM
            'spacy_ner': 0.85,
            'transformers_ner': 0.9,
            'text_frequency': 0.6
        }
        
        # Group by brand and calculate weighted scores
        brand_scores = {}
        
        for result in results:
            brand = result['brand'].upper()
            method = result['method']
            confidence = result['confidence']
            
            # Apply method weight
            weight = method_weights.get(method, 0.7)
            weighted_score = confidence * weight
            
            if brand not in brand_scores:
                brand_scores[brand] = {
                    'scores': [],
                    'methods': [],
                    'results': []
                }
            
            brand_scores[brand]['scores'].append(weighted_score)
            brand_scores[brand]['methods'].append(method)
            brand_scores[brand]['results'].append(result)
        
        # Calculate final scores with AI bonuses
        final_scores = {}
        for brand, data in brand_scores.items():
            scores = data['scores']
            methods = data['methods']
            
            # Base score: weighted average
            base_score = sum(scores) / len(scores)
            
            # Bonus for multiple methods agreeing
            method_bonus = min(0.1 * (len(scores) - 1), 0.3)
            
            # Extra bonus for AI methods
            ai_methods = ['openai_llm', 'spacy_ner', 'transformers_ner']
            ai_bonus = 0.05 * len([m for m in methods if m in ai_methods])
            
            final_score = min(base_score + method_bonus + ai_bonus, 1.0)
            
            final_scores[brand] = {
                'score': final_score,
                'methods': methods,
                'results': data['results'],
                'num_methods': len(scores)
            }
        
        # Get best brand
        best_brand = max(final_scores.keys(), key=lambda x: final_scores[x]['score'])
        best_data = final_scores[best_brand]
        
        # Determine primary method
        ai_methods_used = [m for m in best_data['methods'] if m in ['openai_llm', 'spacy_ner', 'transformers_ner']]
        if ai_methods_used:
            primary_method = f"ai_ensemble_{len(ai_methods_used)}_methods"
        else:
            primary_method = "traditional_ensemble"
        
        return {
            'brand': best_brand.title(),
            'confidence': best_data['score'],
            'method': primary_method,
            'details': {
                'contributing_methods': best_data['methods'],
                'num_agreeing_methods': best_data['num_methods'],
                'ai_methods_used': ai_methods_used,
                'all_results': best_data['results'],
                'all_candidates': [brand.title() for brand in final_scores.keys()]
            }
        }