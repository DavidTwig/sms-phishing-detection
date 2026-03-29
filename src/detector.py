"""
SMS Phishing Detector using Llama 3.3 via Groq API.
Based on SmishX methodology with context gathering and confidence scoring.
Uses a two-pass classification system for improved accuracy.
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
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def extract_urls(self, text: str) -> list:
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
        result = {
            'original_url': url, 'final_url': url,
            'redirect_chain': [], 'num_redirects': 0, 'error': None
        }
        try:
            response = requests.head(url, allow_redirects=True, timeout=self.timeout, headers=self.headers)
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
        result = {
            'domain': None, 'creation_date': None,
            'registrar': None, 'domain_age_days': None, 'error': None
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
        result = {
            'url': url, 'title': None, 'text_content': None,
            'forms_detected': False, 'password_field': False, 'error': None
        }
        try:
            response = requests.get(url, timeout=self.timeout, headers=self.headers, verify=False)
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
        context = {'urls_found': [], 'url_analyses': []}
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
# SYSTEM PROMPTS - FIRST PASS (classification)
# ============================================================
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

# ============================================================
# SYSTEM PROMPT - SECOND PASS (legitimacy review)
# ============================================================
SECOND_PASS_SYSTEM_PROMPT = """You are a message authenticity reviewer. A security system has flagged the following SMS message as potentially suspicious (classified as {first_classification}). Your job is to determine whether this might actually be a LEGITIMATE message from a real service.

Many genuine messages look suspicious because they contain URLs, brand names, urgency language, or requests to take action. These are common in real notifications from companies.

Signs this could be a GENUINE legitimate message:
- Specific order numbers, tracking codes, or reference IDs (e.g., "order ID OD222900438837406000")
- One-time passwords (OTP) or verification codes with specific numbers
- Transaction alerts with specific amounts and card numbers (e.g., "Rs. 2,500 on card ending 4821")
- Delivery status updates referencing specific items or dates
- Feedback requests from a service the user may have recently used
- App notifications (WhatsApp verification, food delivery updates, ride confirmations)
- Data usage or account balance notifications from a telecom provider
- Appointment reminders with specific times and locations

Signs this is genuinely suspicious (NOT legitimate):
- No specific details (vague "your account", "your package" with no specifics)
- URL domain does not match the claimed brand
- Asks for passwords, PINs, or full card numbers
- Threatens account closure or legal action
- Offers prizes, rewards, or money for clicking a link

Based on these criteria, is this message actually legitimate?

Respond in EXACTLY this format:
VERDICT: [legitimate/suspicious]
CONFIDENCE: [0-100]
EXPLANATION: [2-3 sentences explaining why]"""


class SMSPhishingDetector:
    """
    Detects SMS phishing using LLM-based analysis with two-pass classification.
    Pass 1: Standard classification (legitimate/spam/smishing)
    Pass 2: For messages classified as spam/smishing, review whether they
            might actually be legitimate notifications.
    """
    
    def __init__(self, use_context=True, use_two_pass=True):
        """
        Initialise the detector with Groq client.
        
        Args:
            use_context: Whether to gather URL context (slower but more accurate)
            use_two_pass: Whether to use second-pass legitimacy review
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        self.use_context = use_context
        self.use_two_pass = use_two_pass
        self.context_gatherer = ContextGatherer() if use_context else None
    
    def _first_pass(self, sms_text: str, context_data=None, context_text="") -> dict:
        """First pass: standard classification."""
        if self.use_context and context_data and context_data['urls_found']:
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
            temperature=0.1
        )
        
        response_text = response.choices[0].message.content
        result = self._parse_response(response_text)
        result['raw_response'] = response_text
        return result
    
    def _second_pass(self, sms_text: str, first_classification: str) -> dict:
        """Second pass: review whether a flagged message is actually legitimate."""
        system_msg = SECOND_PASS_SYSTEM_PROMPT.format(first_classification=first_classification)
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
            temperature=0.1
        )
        
        response_text = response.choices[0].message.content
        return self._parse_second_pass_response(response_text)
    
    def detect(self, sms_text: str) -> dict:
        """
        Analyse an SMS message for phishing using two-pass classification.
        """
        # Gather context if enabled
        context_text = ""
        context_data = None
        
        if self.use_context:
            context_data = self.context_gatherer.gather_context(sms_text)
            context_text = self.context_gatherer.format_context_for_prompt(context_data)
        
        # First pass: standard classification
        result = self._first_pass(sms_text, context_data, context_text)
        
        # Second pass: if classified as spam or smishing, check if actually legitimate
        if self.use_two_pass and result['classification'] in ['spam', 'smishing']:
            second_result = self._second_pass(sms_text, result['classification'])
            
            if second_result['verdict'] == 'legitimate':
                # Override: the message is actually legitimate
                result['original_classification'] = result['classification']
                result['original_confidence'] = result['confidence']
                result['classification'] = 'legitimate'
                result['confidence'] = second_result['confidence']
                result['explanation'] = f"Second-pass review: {second_result['explanation']}"
                result['two_pass_override'] = True
            else:
                result['two_pass_override'] = False
        
        if self.use_context and context_data:
            result['context'] = context_data
        
        return result
    
    def _parse_response(self, response_text: str) -> dict:
        """Parse first pass LLM response."""
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
    
    def _parse_second_pass_response(self, response_text: str) -> dict:
        """Parse second pass legitimacy review response."""
        result = {
            'verdict': 'suspicious',
            'confidence': 0,
            'explanation': ''
        }
        lines = response_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.upper().startswith('VERDICT:'):
                verdict = line.split(':', 1)[1].strip().lower()
                if verdict in ['legitimate', 'suspicious']:
                    result['verdict'] = verdict
            elif line.upper().startswith('CONFIDENCE:'):
                try:
                    confidence = int(line.split(':', 1)[1].strip().replace('%', ''))
                    result['confidence'] = max(0, min(100, confidence))
                except ValueError:
                    result['confidence'] = 50
            elif line.upper().startswith('EXPLANATION:'):
                result['explanation'] = line.split(':', 1)[1].strip()
        return result