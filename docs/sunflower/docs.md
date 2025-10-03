# Sunflower API Documentation

**Version:** 1.0  
**Date:** September 2025  
**Developed by:** Sunbird AI  

---

## Table of Contents

1. [Introduction](#introduction)
2. [Authentication](#authentication)
3. [Rate Limiting](#rate-limiting)
4. [API Endpoints](#api-endpoints)
5. [Error Handling](#error-handling)
6. [Code Examples](#code-examples)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Introduction

The Sunflower API provides access to Sunbird AI's multilingual language model, specifically designed for Ugandan languages. The API specializes in:

- **Multilingual conversations** in Ugandan languages (Luganda, Acholi, Ateso, etc.)
- **Cross-lingual translations** and explanations
- **Cultural context understanding**
- **Educational content** in local languages

### Key Features

- Automatic retry with exponential backoff
- Context-aware responses
- Usage tracking and monitoring
- Support for custom system messages
- Message history management
- Professional error handling

### Model Information

- **Primary Model:** Sunbird/Sunflower-14B-FP8
- **Supported Languages:** Luganda, Acholi, Ateso, English, and other Ugandan languages
- **Model Types:** Qwen (default), Gemma

---

## Authentication

All API requests require authentication using a valid API key. Include your API key in the request headers:

```http
Authorization: Bearer YOUR_API_KEY
```

### Obtaining API Keys
1. If you don't already have an account, create one at https://api.sunbird.ai/register and login.
2. Go to the [tokens page](https://api.sunbird.ai/tokens) to get your access token which you'll use to authenticate
---

## Rate Limiting

The API implements rate limiting based on account types:

- **Free Tier:** Limited requests per hour
- **Professional:** Higher rate limits
- **Enterprise:** Custom rate limits

Rate limit information is included in response headers:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Time when the rate limit resets

---

## API Endpoints

### 1. Chat Completions Endpoint

**Endpoint:** `POST /ug40_inference`

Professional endpoint for multilingual chat completions with full conversation management.

#### Request Format

```json
{
  "messages": [
    {
      "role": "system|user|assistant",
      "content": "Message content"
    }
  ],
  "model_type": "qwen",
  "temperature": 0.3,
  "stream": false,
  "system_message": "Optional custom system message"
}
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `messages` | Array | Yes | - | List of conversation messages |
| `model_type` | String | No | "qwen" | Model type: "qwen" |
| `temperature` | Float | No | 0.3 | Sampling temperature (0.0-2.0) |
| `stream` | Boolean | No | false | Whether to stream the response |
| `system_message` | String | No | null | Custom system message |

#### Message Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | String | Yes | Message role: "system", "user", or "assistant" |
| `content` | String | Yes | Message content (cannot be empty) |

#### Response Format

```json
{
  "content": "AI response content",
  "model_type": "qwen",
  "usage": {
    "completion_tokens": 150,
    "prompt_tokens": 100,
    "total_tokens": 250
  },
  "processing_time": 2.35,
  "inference_time": 1.85,
  "message_count": 3
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `content` | String | The AI's response |
| `model_type` | String | Model used for inference |
| `usage` | Object | Token usage statistics |
| `processing_time` | Float | Total processing time in seconds |
| `inference_time` | Float | Model inference time in seconds |
| `message_count` | Integer | Number of messages processed |

### 2. Simple Inference Endpoint

**Endpoint:** `POST /ug40_simple`

Simplified interface for single instruction/response interactions.

#### Request Format (Form Data)

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `instruction` | String | Yes | - | The instruction or question |
| `model_type` | String | No | "qwen" | Model type: "qwen" |
| `temperature` | Float | No | 0.3 | Sampling temperature (0.0-2.0) |
| `system_message` | String | No | null | Custom system message |

#### Response Format

```json
{
  "response": "AI response content",
  "model_type": "qwen",
  "processing_time": 1.85,
  "usage": {
    "completion_tokens": 120,
    "prompt_tokens": 80,
    "total_tokens": 200
  },
  "success": true
}
```

---

## Error Handling

The API uses standard HTTP status codes and provides detailed error messages.

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input |
| 401 | Unauthorized - Invalid API key |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 502 | Bad Gateway - Empty model response |
| 503 | Service Unavailable - Model loading |
| 504 | Gateway Timeout - Request timeout |

### Error Response Format

```json
{
  "detail": "Error description",
  "status_code": 400,
  "timestamp": "2025-09-26T10:30:00Z"
}
```

### Common Error Messages

#### Model Loading (503)
```json
{
  "detail": "The AI model is currently loading. This usually takes 2-3 minutes. Please try again shortly."
}
```

#### Timeout (504)
```json
{
  "detail": "The request timed out. Please try again with a shorter prompt or check your network connection."
}
```

#### Invalid Request (400)
```json
{
  "detail": "Message 0 content cannot be empty"
}
```

---

## Code Examples

### Python Examples

#### 1. Basic Chat Completion

```python
import requests
import json

def chat_with_sunflower(messages, api_key, base_url="https://api.sunbirdai.com"):
    """
    Send a chat completion request to Sunflower API
    """
    url = f"{base_url}/ug40_inference"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": messages,
        "model_type": "qwen",
        "temperature": 0.3,
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

# Example usage
api_key = "your_api_key_here"

messages = [
    {
        "role": "system", 
        "content": "You are Sunflower, a multilingual assistant for Ugandan languages made by Sunbird AI."
    },
    {
        "role": "user", 
        "content": "Translate 'Good morning' to Luganda"
    }
]

result = chat_with_sunflower(messages, api_key)
if result:
    print(f"Response: {result['content']}")
    print(f"Tokens used: {result['usage']['total_tokens']}")
```

#### 2. Simple Inference

```python
import requests

def simple_inference(instruction, api_key, base_url="https://api.sunbirdai.com"):
    """
    Send a simple inference request to Sunflower API
    """
    url = f"{base_url}/ug40_simple"
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "instruction": instruction,
        "model_type": "qwen",
        "temperature": 0.3
    }
    
    try:
        response = requests.post(url, headers=headers, data=data, timeout=30)
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

# Example usage
api_key = "your_api_key_here"
instruction = "Explain the meaning of 'Ubuntu' in Ugandan context"

result = simple_inference(instruction, api_key)
if result:
    print(f"Response: {result['response']}")
```

#### 3. Conversation Management

```python
class SunflowerConversation:
    def __init__(self, api_key, base_url="https://api.sunbirdai.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.messages = []
        
    def set_system_message(self, message):
        """Set or update the system message"""
        # Remove existing system message
        self.messages = [msg for msg in self.messages if msg["role"] != "system"]
        # Add new system message at the beginning
        self.messages.insert(0, {"role": "system", "content": message})
    
    def add_user_message(self, content):
        """Add a user message to the conversation"""
        self.messages.append({"role": "user", "content": content})
    
    def get_response(self, model_type="qwen", temperature=0.3):
        """Get AI response and add it to conversation"""
        url = f"{self.base_url}/ug40_inference"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": self.messages,
            "model_type": model_type,
            "temperature": temperature
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Add assistant response to conversation
            self.messages.append({
                "role": "assistant", 
                "content": result["content"]
            })
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
    
    def clear_conversation(self):
        """Clear conversation history but keep system message"""
        system_messages = [msg for msg in self.messages if msg["role"] == "system"]
        self.messages = system_messages

# Example usage
conversation = SunflowerConversation("your_api_key_here")

# Set system message
conversation.set_system_message(
    "You are Sunflower, a helpful assistant specializing in Ugandan languages and culture."
)

# Have a conversation
conversation.add_user_message("Hello, can you greet me in Luganda?")
response1 = conversation.get_response()
print(f"AI: {response1['content']}")

conversation.add_user_message("How do I say 'thank you' in Acholi?")
response2 = conversation.get_response()
print(f"AI: {response2['content']}")
```

#### 4. Error Handling with Retry Logic

```python
import time
import random
from typing import Optional

def sunflower_request_with_retry(
    messages, 
    api_key, 
    max_retries=3,
    base_delay=2.0,
    base_url="https://api.sunbirdai.com"
) -> Optional[dict]:
    """
    Make a request to Sunflower API with exponential backoff retry
    """
    url = f"{base_url}/ug40_inference"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": messages,
        "model_type": "qwen",
        "temperature": 0.3
    }
    
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            
            elif response.status_code == 503:
                # Model loading - retry with longer delay
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Model loading, retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                    continue
            
            elif response.status_code == 429:
                # Rate limited - retry with exponential backoff
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    print(f"Rate limited, retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                    continue
            
            else:
                print(f"Request failed with status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                print(f"Request timed out, retrying in {delay:.1f} seconds...")
                time.sleep(delay)
                continue
            else:
                print("Request timed out after all retries")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
    
    print("All retry attempts exhausted")
    return None

# Example usage
messages = [
    {"role": "user", "content": "What is the capital of Uganda?"}
]

result = sunflower_request_with_retry(messages, "your_api_key_here")
if result:
    print(f"Response: {result['content']}")
```

### JavaScript/Node.js Examples

#### 1. Basic Chat Completion

```javascript
const axios = require('axios');

async function chatWithSunflower(messages, apiKey, baseUrl = 'https://api.sunbirdai.com') {
    const url = `${baseUrl}/ug40_inference`;
    
    const headers = {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
    };
    
    const payload = {
        messages: messages,
        model_type: 'qwen',
        temperature: 0.3,
        stream: false
    };
    
    try {
        const response = await axios.post(url, payload, { 
            headers: headers,
            timeout: 30000 
        });
        return response.data;
    } catch (error) {
        console.error('Request failed:', error.message);
        if (error.response) {
            console.error('Response data:', error.response.data);
        }
        return null;
    }
}

// Example usage
const apiKey = 'your_api_key_here';

const messages = [
    {
        role: 'system',
        content: 'You are Sunflower, a multilingual assistant for Ugandan languages.'
    },
    {
        role: 'user',
        content: 'How do you say "welcome" in Luganda?'
    }
];

chatWithSunflower(messages, apiKey)
    .then(result => {
        if (result) {
            console.log('Response:', result.content);
            console.log('Tokens used:', result.usage.total_tokens);
        }
    });
```

#### 2. Simple Inference

```javascript
const FormData = require('form-data');
const axios = require('axios');

async function simpleInference(instruction, apiKey, baseUrl = 'https://api.sunbirdai.com') {
    const url = `${baseUrl}/ug40_simple`;
    
    const formData = new FormData();
    formData.append('instruction', instruction);
    formData.append('model_type', 'qwen');
    formData.append('temperature', '0.3');
    
    const headers = {
        'Authorization': `Bearer ${apiKey}`,
        ...formData.getHeaders()
    };
    
    try {
        const response = await axios.post(url, formData, {
            headers: headers,
            timeout: 30000
        });
        return response.data;
    } catch (error) {
        console.error('Request failed:', error.message);
        return null;
    }
}

// Example usage
const apiKey = 'your_api_key_here';
const instruction = 'Translate "How are you?" to Ateso';

simpleInference(instruction, apiKey)
    .then(result => {
        if (result) {
            console.log('Response:', result.response);
        }
    });
```

### cURL Examples

#### 1. Chat Completion

```bash
curl -X POST "https://api.sunbirdai.com/ug40_inference" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "system",
        "content": "You are Sunflower, a multilingual assistant for Ugandan languages made by Sunbird AI."
      },
      {
        "role": "user", 
        "content": "Translate \"Good evening\" to Luganda"
      }
    ],
    "model_type": "qwen",
    "temperature": 0.3
  }'
```

#### 2. Simple Inference

```bash
curl -X POST "https://api.sunbirdai.com/ug40_simple" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "instruction=What are some traditional Ugandan foods?" \
  -F "model_type=qwen" \
  -F "temperature=0.3"
```

---

## Best Practices

### 1. Message Management

- **Always include a system message** to provide context and improve response quality
- **Keep conversation history relevant** - trim old messages to stay within token limits
- **Use clear, specific prompts** for better results

### 2. Error Handling

- **Implement retry logic** for transient errors (503, 504, 429)
- **Handle model loading delays** - allow 2-3 minutes for cold starts
- **Validate input** before sending requests

### 3. Performance Optimization

- **Use appropriate temperature settings**:
  - 0.0-0.3: More deterministic, good for translations
  - 0.4-0.7: Balanced creativity
  - 0.8-1.0: More creative responses
- **Consider using simple endpoint** for single-turn interactions
- **Implement caching** for repeated requests

### 4. Rate Limiting

- **Monitor rate limit headers** in responses
- **Implement exponential backoff** for rate limit exceeded errors
- **Consider request batching** where appropriate

### 5. Security

- **Never expose API keys** in client-side code
- **Use environment variables** to store credentials
- **Implement proper authentication** in your applications

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Model Loading Errors (503)

**Problem:** "The AI model is currently loading"

**Solutions:**
- Wait 2-3 minutes before retrying
- Implement exponential backoff retry logic
- Use longer timeout values during cold starts

#### 2. Empty Responses (502)

**Problem:** "The model returned an empty response"

**Solutions:**
- Rephrase your request to be more specific
- Check if the input contains inappropriate content
- Try a different temperature setting

#### 3. Timeout Errors (504)

**Problem:** Request times out

**Solutions:**
- Reduce prompt length
- Use simpler queries
- Increase timeout values in your code
- Check network connectivity

#### 4. Rate Limiting (429)

**Problem:** Too many requests

**Solutions:**
- Implement request queuing
- Use exponential backoff
- Consider upgrading your account tier

#### 5. Invalid Message Format (400)

**Problem:** Message validation errors

**Solutions:**
- Ensure all messages have 'role' and 'content' fields
- Check that content is not empty
- Validate role values ('system', 'user', 'assistant')

### Debugging Tips

1. **Enable detailed logging** to track request/response cycles
2. **Test with simple queries** first before complex conversations
3. **Use the simple endpoint** to isolate issues
4. **Check API status** at Sunbird AI's status page
5. **Monitor token usage** to avoid unexpected costs

---

## Support and Resources

### Getting Help

- **Email Support:** support@sunbirdai.com
- **Documentation:** https://docs.sunbirdai.com
- **Status Page:** https://status.sunbirdai.com

### Additional Resources

- **Model Documentation:** Technical details about Sunflower models
- **Language Support:** Complete list of supported Ugandan languages
- **Use Cases:** Examples and case studies
- **API Updates:** Changelog and version history

---

**© 2025 Sunbird AI. All rights reserved.**

*This documentation is subject to updates and improvements. Please check the official documentation website for the latest version.*