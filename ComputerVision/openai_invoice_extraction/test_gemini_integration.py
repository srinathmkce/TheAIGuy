#!/usr/bin/env python3
"""
Test script to verify Gemini integration with LangChain
"""

import os
import getpass
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import base64
from PIL import Image
import io

# Load environment variables
load_dotenv()

def test_gemini_integration():
    """Test basic Gemini functionality"""
    print("🧪 Testing Gemini Integration with LangChain")
    print("=" * 50)
    
    # Set up API key if not already set
    if "GOOGLE_API_KEY" not in os.environ:
        print("Please set your GOOGLE_API_KEY environment variable")
        return False
    
    try:
        # Initialize Gemini model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        print("✅ Gemini model initialized successfully")
        
        # Test 1: Simple text query
        print("\n📝 Test 1: Simple text query")
        test_message = HumanMessage(content="Respond with exactly: 'Gemini is working with LangChain'")
        response = llm.invoke([test_message])
        print(f"Response: {response.content}")
        
        if "Gemini is working with LangChain" in response.content:
            print("✅ Text query test passed")
        else:
            print("❌ Text query test failed")
            return False
        
        # Test 2: Multimodal query (text + image)
        print("\n🖼️ Test 2: Multimodal query (text + image)")
        
        # Create a simple test image
        test_image = Image.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        test_image.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        multimodal_message = HumanMessage(
            content=[
                {"type": "text", "text": "What color is this image? Respond with just the color name."},
                {
                    "type": "image_url", 
                    "image_url": f"data:image/jpeg;base64,{image_base64}"
                },
            ]
        )
        
        multimodal_response = llm.invoke([multimodal_message])
        print(f"Multimodal response: {multimodal_response.content}")
        
        if "red" in multimodal_response.content.lower():
            print("✅ Multimodal query test passed")
        else:
            print("❌ Multimodal query test failed")
            return False
        
        print("\n🎉 All tests passed! Gemini integration is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_gemini_integration()
    exit(0 if success else 1)
