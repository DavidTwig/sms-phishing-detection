"""
SMS Phishing Detector using Llama 3.3 via Groq API.
Based on SmishX methodology with context gathering and confidence scoring.
"""

import os
import re
import socket
from datetime import datetime
from urllib.parse import urlparse
from dotenv import load_dotenv
from groq import Groq
import requests
from bs4 import BeautifulSoup

# Load API key
load_dotenv()


class ContextGatherer:
    """
    Gathers external context about URLs found in SMS messages.
    Implements URL redirect chain, WHOIS lookup, and HTML extraction.
    """
    
    def __init__(self, timeout=10):
        """Initialise with request timeout setting."""
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def extract_urls(self, text: str) -> list:
        """
        Extract all URLs from SMS text.
        Handles both http(s):// and common shortened formats.
        """
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text, re.IGNORECASE)
        
        short_pattern = r'(?<![/@])\b([a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:/[^\s<>"{}|\\^`\[\]]*)?)'
        potential_urls = re.findall(short_pattern, text)
        
        for url in potential_urls:
            if '.' in url and not url.startswith('http'):
                if not re.match(r'^[a-z]\.[a-z]\.?$', url, re.IGNORECASE):
                    if not re.match(r'^\d+\.\d+', url):
                        urls.append('http://' + url)
        
        return list(set(urls))
    
    def follow_redirects(self, url: str) -> dict:
        """
        Follow URL redirect chain to find final destination.
        Returns chain of redirects and final URL.
        """
        result = {
            'original_url': url,
            'final_url': url,
            'redirect_chain': [],
            'num_redirects': 0,
            'error': None
        }
        
        try:
            response = requests.head(
                url, 
                allow_redirects=True, 
                timeout=self.timeout,
                headers=self.headers
            )
            
            if response.history:
                result['redirect_chain'] = [r.url for r in response.history]
                result['redirect_chain'].append(response.url)
                result['final_url'] = response.url
                result['num_redirects'] = len(response.history)
            else:
                result['final_url'] = response.url
                
        except requests.exceptions.SSLError:
            result['error'] = 'SSL certificate error (suspicious)'
        except requests.exceptions.ConnectionError:
            result['error'] = 'Connection failed (domain may not exist)'
        except requests.exceptions.Timeout:
            result['error'] = 'Request timed out'
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def get_whois_info(self, url: str) -> dict:
        """
        Get WHOIS information for a domain.
        Returns registration date, registrar, and domain age.
        """
        result = {
            'domain': None,
            'creation_date': None,
            'registrar': None,
            'domain_age_days': None,
            'error': None
        }
        
        try:
            import whois
            
            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path.split('/')[0]
            domain = domain.replace('www.', '')
            result['domain'] = domain
            
            w = whois.whois(domain)
            
            if w.creation_date:
                creation = w.creation_date
                if isinstance(creation, list):
                    creation = creation[0]
                result['creation_date'] = creation.strftime('%Y-%m-%d') if creation else None
                
                if creation:
                    age = datetime.now() - creation
                    result['domain_age_days'] = age.days
            
            if w.registrar:
                result['registrar'] = w.registrar
                
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def extract_html_content(self, url: str, max_length=2000) -> dict:
        """
        Fetch webpage and extract readable text content.
        Limits content length to avoid token limits.
        """
        result = {
            'url': url,
            'title': None,
            'text_content': None,
            'forms_detected': False,
            'password_field': False,
            'error': None
        }
        
        try:
            response = requests.get(
                url, 
                timeout=self.timeout, 
                headers=self.headers,
                verify=False
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            if soup.title:
                result['title'] = soup.title.string.strip() if soup.title.string else None
            
            forms = soup.find_all('form')
            result['forms_detected'] = len(forms) > 0
            
            password_inputs = soup.find_all('input', {'type': 'password'})
            result['password_field'] = len(password_inputs) > 0
            
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            text = soup.get_text(separator=' ', strip=True)
            text = ' '.join(text.split())
            
            if len(text) > max_length:
                text = text[:max_length] + '...'
            
            result['text_content'] = text
            
        except requests.exceptions.SSLError:
            result['error'] = 'SSL certificate error'
        except requests.exceptions.ConnectionError:
            result['error'] = 'Connection failed'
        except requests.exceptions.Timeout:
            result['error'] = 'Request timed out'
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def gather_context(self, sms_text: str) -> dict:
        """
        Main method: gather all context for URLs in an SMS.
        Returns structured context for LLM analysis.
        """
        context = {
            'urls_found': [],
            'url_analyses': []
        }
        
        urls = self.extract_urls(sms_text)
        context['urls_found'] = urls
        
        for url in urls[:3]:
            analysis = {
                'url': url,
                'redirects': self.follow_redirects(url),
                'whois': self.get_whois_info(url),
                'html': self.extract_html_content(url)
            }
            context['url_analyses'].append(analysis)
        
        return context
    
    def format_context_for_prompt(self, context: dict) -> str:
        """
        Format gathered context into readable text for LLM prompt.
        """
        if not context['urls_found']:
            return "No URLs found in this message."
        
        output = []
        output.append(f"URLs found: {len(context['urls_found'])}")
        
        for i, analysis in enumerate(context['url_analyses'], 1):
            output.append(f"\n--- URL {i}: {analysis['url']} ---")
            
            redirects = analysis['redirects']
            if redirects['error']:
                output.append(f"Redirect check: ERROR - {redirects['error']}")
            else:
                output.append(f"Final destination: {redirects['final_url']}")
                output.append(f"Number of redirects: {redirects['num_redirects']}")
            
            whois_info = analysis['whois']
            if whois_info['error']:
                output.append(f"WHOIS: ERROR - {whois_info['error']}")
            else:
                output.append(f"Domain: {whois_info['domain']}")
                if whois_info['creation_date']:
                    output.append(f"Domain created: {whois_info['creation_date']}")
                    if whois_info['domain_age_days'] is not None:
                        if whois_info['domain_age_days'] < 30:
                            output.append(f"Domain age: {whois_info['domain_age_days']} days (VERY NEW - suspicious)")
                        elif whois_info['domain_age_days'] < 365:
                            output.append(f"Domain age: {whois_info['domain_age_days']} days (relatively new)")
                        else:
                            years = whois_info['domain_age_days'] // 365
                            output.append(f"Domain age: ~{years} years (established)")
                if whois_info['registrar']:
                    output.append(f"Registrar: {whois_info['registrar']}")
            
            html_info = analysis['html']
            if html_info['error']:
                output.append(f"Webpage: ERROR - {html_info['error']}")
            else:
                if html_info['title']:
                    output.append(f"Page title: {html_info['title']}")
                if html_info['forms_detected']:
                    output.append("WARNING: Form detected on page")
                if html_info['password_field']:
                    output.append("WARNING: Password input field detected")
                if html_info['text_content']:
                    preview = html_info['text_content'][:500]
                    output.append(f"Page content preview: {preview}...")
        
        return '\n'.join(output)


# ============================================================
# SYSTEM PROMPTS
# ============================================================
# Full-length prompts delivered via system message.
# This is the configuration that achieved 84.2% accuracy.

SYSTEM_PROMPT_WITH_CONTEXT = """You are an SMS phishing detector with access to external URL analysis. You will receive an SMS message and gathered context about any URLs it contains.

Based on the SMS text AND the URL context, determine if the message is:
1. LEGITIMATE - A genuine message from a real organisation or person
2. SPAM - Unsolicited promotional message where the sender is honest about who they are. They want your money or attention (gambling, crypto, adult content, job offers, promotions). The sender does NOT pretend to be someone else.
3. SMISHING - A deceptive message where the sender IMPERSONATES a trusted entity (bank, delivery service, government, tech company) to steal information, money, or install malware. The key indicator is DECEPTION through impersonation.

KEY DISTINCTION - Ask yourself: "Is the sender pretending to be someone they're not?"
- If YES (fake bank alert, fake delivery notice, fake account warning) = SMISHING
- If NO (gambling site promoting gambling, crypto site promoting crypto) = SPAM

Key indicators for SMISHING:
- Impersonates a bank, delivery service (USPS, Royal Mail), or tech company (Apple, Microsoft)
- Claims there's a problem with your account that needs immediate action
- URL domain does NOT match the brand being impersonated
- Asks you to "verify", "confirm", or "update" your details

Key indicators for SPAM:
- Promotes gambling, casino, lottery, crypto, adult content, or job offers
- Sender is honest about what they're selling (even if unsolicited)
- No impersonation of a different trusted brand

Respond in EXACTLY this format:
CLASSIFICATION: [legitimate/spam/smishing]
CONFIDENCE: [0-100]
EXPLANATION: [2-3 sentences explaining why, referencing the URL context if relevant]"""

SYSTEM_PROMPT_NO_CONTEXT = """You are an SMS phishing detector. You will receive an SMS message and must determine if it is:
1. LEGITIMATE - A genuine message from a real organisation or person
2. SPAM - Unsolicited promotional message where the sender is honest about who they are. They want your money or attention (gambling, crypto, adult content, job offers, promotions). The sender does NOT pretend to be someone else.
3. SMISHING - A deceptive message where the sender IMPERSONATES a trusted entity (bank, delivery service, government, tech company) to steal information, money, or install malware. The key indicator is DECEPTION through impersonation.

KEY DISTINCTION - Ask yourself: "Is the sender pretending to be someone they're not?"
- If YES (fake bank alert, fake delivery notice, fake account warning) = SMISHING
- If NO (gambling site promoting gambling, crypto site promoting crypto) = SPAM

Think step by step:
1. Is the sender impersonating a trusted brand (bank, delivery service, tech company)?
2. Does it claim there's a problem with an account that needs urgent action?
3. Is it just promoting a service (gambling, crypto, jobs) without deception?
4. Does the message create urgency through fear of losing access to something?

Respond in EXACTLY this format:
CLASSIFICATION: [legitimate/spam/smishing]
CONFIDENCE: [0-100]
EXPLANATION: [2-3 sentences explaining why]"""


class SMSPhishingDetector:
    """
    Detects SMS phishing using LLM-based analysis with context gathering.
    Provides classification, confidence score, and explanation.
    """
    
    def __init__(self, use_context=True):
        """
        Initialise the detector with Groq client.
        
        Args:
            use_context: Whether to gather URL context (slower but more accurate)
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        self.use_context = use_context
        self.context_gatherer = ContextGatherer() if use_context else None
    
    def detect(self, sms_text: str) -> dict:
        """
        Analyse an SMS message for phishing.
        
        Args:
            sms_text: The SMS message to analyse
            
        Returns:
            Dictionary with:
                - classification: 'legitimate', 'spam', or 'smishing'
                - confidence: 0-100 score
                - explanation: Why this classification was made
                - context: Gathered URL context (if enabled)
        """
        
        # Gather context if enabled
        context_text = ""
        context_data = None
        
        if self.use_context:
            context_data = self.context_gatherer.gather_context(sms_text)
            context_text = self.context_gatherer.format_context_for_prompt(context_data)
        
        # Build messages using system/user split
        if self.use_context and context_data['urls_found']:
            system_msg = SYSTEM_PROMPT_WITH_CONTEXT
            user_msg = f"""SMS Message:
"{sms_text}"

URL Analysis Context:
{context_text}"""
        else:
            system_msg = SYSTEM_PROMPT_NO_CONTEXT
            user_msg = f"""SMS Message:
"{sms_text}"
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=400,
            temperature=0.1  # Low temperature for consistent results
        )
        
        # Parse the response
        response_text = response.choices[0].message.content
        result = self._parse_response(response_text)
        result['raw_response'] = response_text
        
        if self.use_context:
            result['context'] = context_data
        
        return result
    
    def _parse_response(self, response_text: str) -> dict:
        """Parse the LLM response into structured data."""
        
        result = {
            'classification': 'unknown',
            'confidence': 0,
            'explanation': ''
        }
        
        lines = response_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith('CLASSIFICATION:'):
                classification = line.split(':', 1)[1].strip().lower()
                if classification in ['legitimate', 'spam', 'smishing']:
                    result['classification'] = classification
            elif line.upper().startswith('CONFIDENCE:'):
                try:
                    confidence = int(line.split(':', 1)[1].strip().replace('%', ''))
                    result['confidence'] = max(0, min(100, confidence))
                except ValueError:
                    result['confidence'] = 50
            elif line.upper().startswith('EXPLANATION:'):
                result['explanation'] = line.split(':', 1)[1].strip()
        
        return result