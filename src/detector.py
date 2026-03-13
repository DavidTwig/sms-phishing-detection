"""
SMS Phishing Detector using Llama 3.1 8B via Groq API.
Based on SmishX methodology with added confidence scoring.

Using llama-3.1-8b-instant for higher free tier limits (500K TPD vs 100K TPD).
"""

import os
from dotenv import load_dotenv
from groq import Groq

# Load API key
load_dotenv()

class SMSPhishingDetector:
    """
    Detects SMS phishing using LLM-based analysis.
    Provides classification, confidence score, and explanation.
    """
    
    def __init__(self):
        """Initialise the detector with Groq client."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
    
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
        """
        
        prompt = f"""You are an SMS phishing detector. Analyse the following SMS message and determine if it is:
1. LEGITIMATE - A genuine message from a real organisation or person
2. SPAM - Unsolicited promotional message (gambling, crypto, adult content, etc.)
3. SMISHING - A phishing attempt trying to steal information or trick the user

SMS Message:
"{sms_text}"

Respond in EXACTLY this format:
CLASSIFICATION: [legitimate/spam/smishing]
CONFIDENCE: [0-100]
EXPLANATION: [2-3 sentences explaining why]

Think step by step:
1. Check if the message creates urgency or fear
2. Check if it asks for personal information
3. Check if it contains suspicious links
4. Check if it impersonates a known brand
5. Check if it promotes high-risk services (gambling, crypto, etc.)
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.1  # Low temperature for consistent results
        )
        
        # Parse the response
        response_text = response.choices[0].message.content
        result = self._parse_response(response_text)
        result['raw_response'] = response_text
        
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


# Test the detector
if __name__ == "__main__":
    detector = SMSPhishingDetector()
    
    # Test messages
    test_messages = [
        "Hi mum, can you pick me up from school at 3pm?",
        "URGENT: Your bank account has been compromised! Click here to verify: http://dodgy-link.com",
        "Congratulations! You've won £1000 in our lottery! Claim now: bit.ly/win123",
        "Your Amazon order #12345 has shipped. Track at amazon.com/track",
        "FREE crypto coins! Join now and get 500 FREE Bitcoin: crypto-scam.net"
    ]
    
    print("=" * 60)
    print("SMS PHISHING DETECTOR TEST")
    print(f"Model: llama-3.1-8b-instant (500K TPD free tier)")
    print("=" * 60)
    
    for sms in test_messages:
        print(f"\nSMS: {sms[:60]}...")
        result = detector.detect(sms)
        print(f"  Classification: {result['classification'].upper()}")
        print(f"  Confidence: {result['confidence']}%")
        print(f"  Explanation: {result['explanation']}")
        print("-" * 60)