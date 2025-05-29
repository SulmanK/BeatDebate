"""
Shared LLM Utilities for BeatDebate Agents

Consolidates LLM calling and JSON parsing patterns that are duplicated across agents,
providing a single source of truth for LLM interactions.
"""

import json
import re
from typing import Dict, Any, Optional, Union
import structlog

logger = structlog.get_logger(__name__)


class LLMUtils:
    """
    Shared utilities for LLM interactions across all agents.
    
    Consolidates:
    - LLM calling patterns
    - JSON response parsing
    - Error handling
    - Response cleaning and validation
    """
    
    def __init__(self, llm_client):
        """
        Initialize LLM utilities with client.
        
        Args:
            llm_client: LLM client (e.g., Gemini client)
        """
        self.llm_client = llm_client
        self.logger = logger.bind(component="LLMUtils")
    
    async def call_llm_with_json_response(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Call LLM and parse JSON response with robust error handling.
        
        Args:
            user_prompt: User prompt for the LLM
            system_prompt: System prompt (optional)
            max_retries: Maximum number of retry attempts
            
        Returns:
            Parsed JSON response as dictionary
            
        Raises:
            ValueError: If JSON parsing fails after all retries
            RuntimeError: If LLM call fails
        """
        for attempt in range(max_retries + 1):
            try:
                # Make LLM call
                response_text = await self._make_llm_call(user_prompt, system_prompt)
                
                # Parse JSON response
                json_data = self._parse_json_response(response_text)
                
                self.logger.debug(
                    "LLM JSON response parsed successfully",
                    attempt=attempt + 1,
                    response_keys=list(json_data.keys()) if isinstance(json_data, dict) else None
                )
                
                return json_data
                
            except json.JSONDecodeError as e:
                self.logger.warning(
                    "JSON parsing failed",
                    attempt=attempt + 1,
                    error=str(e),
                    response_preview=response_text[:200] if 'response_text' in locals() else None
                )
                
                if attempt == max_retries:
                    # Try alternative parsing methods on final attempt
                    try:
                        return self._aggressive_json_parsing(response_text)
                    except Exception:
                        raise ValueError(f"Failed to parse JSON after {max_retries + 1} attempts: {e}")
                        
            except Exception as e:
                self.logger.error(
                    "LLM call failed",
                    attempt=attempt + 1,
                    error=str(e)
                )
                
                if attempt == max_retries:
                    raise RuntimeError(f"LLM call failed after {max_retries + 1} attempts: {e}")
        
        # This should never be reached, but just in case
        raise RuntimeError("Unexpected error in LLM call loop")
    
    async def call_llm(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Call LLM and return raw text response.
        
        Args:
            user_prompt: User prompt for the LLM
            system_prompt: System prompt (optional)
            
        Returns:
            Raw LLM response text
            
        Raises:
            RuntimeError: If LLM call fails
        """
        try:
            response_text = await self._make_llm_call(user_prompt, system_prompt)
            
            self.logger.debug(
                "LLM text response received",
                response_length=len(response_text)
            )
            
            return response_text
            
        except Exception as e:
            self.logger.error("LLM call failed", error=str(e))
            raise RuntimeError(f"LLM call failed: {e}")
    
    async def _make_llm_call(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Make actual LLM call with unified error handling.
        
        Args:
            user_prompt: User prompt
            system_prompt: System prompt
            
        Returns:
            LLM response text
        """
        if not self.llm_client:
            raise RuntimeError("LLM client not initialized")
        
        try:
            # Combine system and user prompts
            full_prompt = (
                f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
            )
            
            self.logger.debug(
                "Making LLM call",
                prompt_length=len(full_prompt),
                has_system_prompt=system_prompt is not None
            )
            
            # Call LLM - handle both sync and async clients
            response = self.llm_client.generate_content(full_prompt)
            
            # If it's a coroutine (async client), await it
            if hasattr(response, '__await__'):
                response = await response
            
            return response.text
            
        except Exception as e:
            self.logger.error("LLM API call failed", error=str(e))
            raise
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON response with robust error handling and cleaning.
        
        Args:
            response_text: Raw LLM response text
            
        Returns:
            Parsed JSON data
            
        Raises:
            json.JSONDecodeError: If JSON parsing fails
        """
        try:
            # Clean the response text
            cleaned_text = self._clean_response_text(response_text)
            
            # Extract JSON boundaries
            json_str = self._extract_json_boundaries(cleaned_text)
            
            # Additional JSON cleaning for common LLM issues
            json_str = self._clean_json_string(json_str)
            
            # Parse JSON
            json_data = json.loads(json_str)
            
            self.logger.debug(
                "JSON parsing successful",
                original_length=len(response_text),
                cleaned_length=len(json_str),
                keys=list(json_data.keys()) if isinstance(json_data, dict) else None
            )
            
            return json_data
            
        except json.JSONDecodeError as e:
            self.logger.warning(
                "Initial JSON parsing failed",
                error=str(e),
                response_preview=response_text[:300]
            )
            raise
    
    def _clean_response_text(self, response_text: str) -> str:
        """Clean response text by removing markdown and explanatory text."""
        cleaned = response_text.strip()
        
        # Remove markdown code blocks
        if cleaned.startswith('```'):
            lines = cleaned.split('\n')
            # Remove first line if it's markdown
            if lines[0].startswith('```'):
                lines = lines[1:]
            # Remove last line if it's markdown
            if lines and lines[-1].startswith('```'):
                lines = lines[:-1]
            cleaned = '\n'.join(lines)
        
        return cleaned.strip()
    
    def _extract_json_boundaries(self, text: str) -> str:
        """Extract JSON object boundaries from text."""
        # Find the first opening brace
        start_idx = text.find('{')
        if start_idx == -1:
            raise ValueError("No JSON object found in response")
        
        # Find matching closing brace by counting braces
        brace_count = 0
        end_idx = start_idx
        
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if brace_count != 0:
            # If braces don't match, try to find the last closing brace
            end_idx = text.rfind('}')
            if end_idx == -1 or end_idx <= start_idx:
                raise ValueError("Unmatched braces in JSON response")
            end_idx += 1
        
        return text[start_idx:end_idx]
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean JSON string to fix common LLM formatting issues."""
        # Remove any trailing commas before closing braces/brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Remove any comments (// or /* */)
        json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # Fix common typos in boolean/null values
        json_str = re.sub(r'\btrue\b', 'true', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r'\bfalse\b', 'false', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r'\bnull\b', 'null', json_str, flags=re.IGNORECASE)
        
        # Replace single quotes with double quotes for keys and string values
        # This is a simple approach - for complex cases, we'd need a proper parser
        json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)  # Keys
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)  # String values
        
        return json_str
    
    def _aggressive_json_parsing(self, response_text: str) -> Dict[str, Any]:
        """
        Aggressive JSON parsing as a last resort.
        
        Args:
            response_text: Raw response text
            
        Returns:
            Parsed JSON data
            
        Raises:
            ValueError: If all parsing attempts fail
        """
        self.logger.info("Attempting aggressive JSON parsing")
        
        # Attempt 1: Try fixing common JSON issues
        try:
            fixed_json = self._fix_common_json_issues(response_text)
            return json.loads(fixed_json)
        except Exception as e:
            self.logger.debug("Fixed JSON parsing failed", error=str(e))
        
        # Attempt 2: Use regex to extract JSON-like structure
        try:
            extracted_json = self._extract_json_with_regex(response_text)
            if extracted_json:
                return json.loads(extracted_json)
        except Exception as e:
            self.logger.debug("Regex JSON extraction failed", error=str(e))
        
        # Attempt 3: Try to build JSON from key-value patterns
        try:
            constructed_json = self._construct_json_from_patterns(response_text)
            if constructed_json:
                return constructed_json
        except Exception as e:
            self.logger.debug("Pattern-based JSON construction failed", error=str(e))
        
        raise ValueError("All aggressive JSON parsing attempts failed")
    
    def _fix_common_json_issues(self, response_text: str) -> str:
        """Attempt to fix common JSON formatting issues."""
        # Find JSON boundaries more aggressively
        start_idx = response_text.find('{')
        if start_idx == -1:
            return response_text
        
        # Extract everything from first { to last }
        end_idx = response_text.rfind('}')
        if end_idx == -1:
            return response_text
        
        json_candidate = response_text[start_idx:end_idx + 1]
        
        # Apply aggressive cleaning
        json_candidate = self._clean_json_string(json_candidate)
        
        # Remove any text before first { or after last }
        json_candidate = re.sub(r'^[^{]*', '', json_candidate)
        json_candidate = re.sub(r'}[^}]*$', '}', json_candidate)
        
        return json_candidate
    
    def _extract_json_with_regex(self, response_text: str) -> Optional[str]:
        """Extract JSON using regex patterns as a last resort."""
        # Look for JSON-like structure with balanced braces
        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        if matches:
            # Return the longest match (most likely to be complete)
            longest_match = max(matches, key=len)
            return self._clean_json_string(longest_match)
        
        return None
    
    def _construct_json_from_patterns(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Construct JSON from key-value patterns in text."""
        try:
            # Look for key-value patterns like "key": "value" or "key": value
            kv_pattern = r'"([^"]+)":\s*(?:"([^"]*)"|([^,}\s]+))'
            matches = re.findall(kv_pattern, response_text)
            
            if matches:
                result = {}
                for key, str_value, other_value in matches:
                    value = str_value if str_value else other_value
                    
                    # Try to convert to appropriate type
                    if value.lower() == 'true':
                        result[key] = True
                    elif value.lower() == 'false':
                        result[key] = False
                    elif value.lower() == 'null':
                        result[key] = None
                    elif value.isdigit():
                        result[key] = int(value)
                    elif self._is_float(value):
                        result[key] = float(value)
                    else:
                        result[key] = value
                
                return result if result else None
        except Exception as e:
            self.logger.debug("Pattern-based JSON construction failed", error=str(e))
        
        return None
    
    def _is_float(self, value: str) -> bool:
        """Check if string represents a float."""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def validate_json_structure(
        self,
        json_data: Dict[str, Any],
        required_keys: Optional[list] = None,
        optional_keys: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Validate and enhance JSON structure.
        
        Args:
            json_data: Parsed JSON data
            required_keys: List of required keys
            optional_keys: List of optional keys to set defaults for
            
        Returns:
            Validated and enhanced JSON data
        """
        if not isinstance(json_data, dict):
            raise ValueError("JSON data must be a dictionary")
        
        # Check required keys
        if required_keys:
            missing_keys = [key for key in required_keys if key not in json_data]
            if missing_keys:
                self.logger.warning("Missing required keys", missing_keys=missing_keys)
                # Set default values for missing required keys
                for key in missing_keys:
                    json_data[key] = self._get_default_value_for_key(key)
        
        # Set defaults for optional keys
        if optional_keys:
            for key in optional_keys:
                if key not in json_data:
                    json_data[key] = self._get_default_value_for_key(key)
        
        self.logger.debug(
            "JSON structure validated",
            keys=list(json_data.keys()),
            required_keys=required_keys,
            optional_keys=optional_keys
        )
        
        return json_data
    
    def _get_default_value_for_key(self, key: str) -> Union[str, list, dict, int, float]:
        """Get appropriate default value based on key name."""
        # Common key patterns and their default values
        if 'list' in key.lower() or key.endswith('s'):
            return []
        elif 'dict' in key.lower() or 'entities' in key.lower():
            return {}
        elif 'count' in key.lower() or 'score' in key.lower():
            return 0
        elif 'confidence' in key.lower():
            return 0.0
        elif 'intent' in key.lower():
            return 'discovery'
        elif 'complexity' in key.lower():
            return 'medium'
        else:
            return ""
    
    def create_structured_prompt(
        self,
        task_description: str,
        input_data: Dict[str, Any],
        output_format: Dict[str, Any],
        examples: Optional[list] = None
    ) -> str:
        """
        Create a structured prompt for consistent LLM interactions.
        
        Args:
            task_description: Description of the task
            input_data: Input data to include in prompt
            output_format: Expected output format
            examples: Optional examples to include
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            f"Task: {task_description}",
            "",
            "Input Data:",
            json.dumps(input_data, indent=2),
            "",
            "Required Output Format:",
            json.dumps(output_format, indent=2),
            ""
        ]
        
        if examples:
            prompt_parts.extend([
                "Examples:",
                *[json.dumps(example, indent=2) for example in examples],
                ""
            ])
        
        prompt_parts.extend([
            "Instructions:",
            "- Return ONLY the JSON object with no additional text",
            "- Ensure all required fields are included",
            "- Use the exact format specified above",
            ""
        ])
        
        return "\n".join(prompt_parts) 