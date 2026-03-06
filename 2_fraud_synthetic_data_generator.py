import boto3
import json
import logging
from botocore.config import Config
import csv
import io

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- AWS Configuration ---
REGION = 'us-west-2' 

# --- AWS Clients ---
s3 = boto3.client('s3')

# Custom config for Bedrock to handle longer running requests
custom_config = Config(
    read_timeout=1500, 
    connect_timeout=2500,
    retries={'max_attempts': 0}
)
bedrock_runtime = boto3.client('bedrock-runtime', region_name=REGION, config=custom_config)

# --- CRITICAL HELPER FUNCTION ---

def build_agent_response(action_group, function_name, message, response_state):
    """
    Helper function to build the Bedrock Agent compliant response JSON.
    """
    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": action_group,
            "function": function_name,
            "functionResponse": {
                "responseState": response_state, 
                "responseBody": {
                    "TEXT": {
                        "body": message
                    }
                }
            }
        }
    }

# --- Other Helper Functions ---

def get_llm_response(prompt: str) -> str:
    model_id = 'arn:aws:bedrock:us-west-2:640237069320:inference-profile/global.anthropic.claude-sonnet-4-5-20250929-v1:0'
    
    # UPDATED: Increased max_tokens to 4096 to prevent output truncation
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50000, 
        "temperature": 0.0,
        "anthropic_version": "bedrock-2023-05-31"
    })
    
    response = bedrock_runtime.invoke_model(
        modelId=model_id, body=body, contentType='application/json', accept='application/json'
    )
    response_body = json.loads(response['body'].read())
    return response_body.get('content', [{}])[0].get('text', 'LLM did not return text.')

def get_s3_file_content(bucket: str, key: str) -> str:
    response = s3.get_object(Bucket=bucket, Key=key)
    return response['Body'].read().decode('utf-8')

# --- Main Lambda Handler ---

def process_to_csv(llm_text: str) -> str:
    """
    Extracts the Markdown table from LLM text and converts it to a standard CSV string.
    """
    lines = llm_text.split('\n')
    # Filter for lines that look like table rows
    table_rows = [line.strip() for line in lines if line.strip().startswith('|')]
    
    if not table_rows:
        return ""

    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

    for row in table_rows:
        # Skip the Markdown separator line (e.g., |---|---|)
        if all(c in '|- ' for c in row):
            continue
        
        # Split by pipe, strip whitespace, and remove empty strings from ends
        columns = [col.strip() for col in row.split('|') if col.strip() != '']
        if columns:
            writer.writerow(columns)

    return output.getvalue()


def lambda_handler(event, context):
    logger.info("Event received: " + json.dumps(event))

    action_group = event.get('actionGroup')
    function_name = event.get('function')
    
    # --- 1. Agent Input Retrieval ---
    try:
        if not action_group or not function_name:
            raise KeyError("Missing actionGroup or function name in event.")

        params = {p['name']: p['value'] for p in event.get('parameters', [])}
        
        if function_name != 'fraud_detection_synthetic_function':
             raise ValueError(f"Unknown function: {function_name}")
             
        bucket_name = params['bucket_name']
        key_new_document = params['new_document_key'] 
        
        model_agent_evaluation_output = params['model_agent_evaluation_output_file']
        transaction_data = params['traintest_data']

        output_folder = params.get('output_folder', 'updated_output/') 
        
    except (KeyError, ValueError, TypeError) as e:
        error_message = f'Missing or invalid Agent parameter. Details: {str(e)}'
        return build_agent_response(action_group, function_name, error_message, response_state="REPROMPT")

    try:
        logger.info(f"Bucket: {bucket_name}, New Doc Key: {key_new_document}")

        file_content = get_s3_file_content(bucket_name, key_new_document)
        model_agent_evaluation_file = get_s3_file_content(bucket_name, model_agent_evaluation_output)
        traintest_file = get_s3_file_content(bucket_name, transaction_data)
        
        logger.info("Successfully read file contents.")
        
        # 4. Construct the prompt 
        # UPDATED: Prompt logic to ensure ALL users are included and token limits aren't hit.
        
        prompt_template = f"""
            Human:
            You are the Synthetic Data Generator Agent.
            
            Input Data Sources:
            <new_document>
            {file_content} 
            </new_document>
            
            <model_agent_evaluation_output>
            {model_agent_evaluation_file}
            </model_agent_evaluation_output>
            
            <training_data_reference>
            {traintest_file}
            </training_data_reference>

            CRITICAL OBJECTIVE: 
            You must generate synthetic records for EVERY SINGLE unique user_id found in <new_document>. 
            Do NOT summarize. Do NOT skip any user_id. The output table must be complete.

            Objectives:
                1) Learn patterns from the inputs.
                2) Generate synthetic records for EVERY unique user_id using TWO approaches:
                    (A) Generative Models (GAN/VAE/LLM-like behavior)
            
            Constraints:
                - Do NOT copy exact PII. All IDs/IPs must be synthetic.
                - Maintain plausible distributions.
                - Field names MUST be EXACTLY: user_id, card_id, transaction_id, amount, merchant_category, channel, country_risk, time_of_day, customer_tenure_months, is_international, velocity_1h, is_fraud, cardtype, ip_address, card_age, card_30days_transaction_count, past_1hr_transaction_count, country_name, timestamp.
                  
                    merchant_category: PHARMA, FUEL, JEWELRY, RESTAURANT, GROCERY, TRAVEL, ONLINE_MARKET, or ELECTRONICS.
                    channel: ONLINE, ATM, POS, or MOBILE.
                    country_risk (e.g., Low, Medium, High)
                    time_of_day (e.g., Morning, Afternoon, Evening, Night)
                    is_international: 1 or 0.
                    cardtype (e.g., Credit, Debit, Prepaid)
                    country_name: United States, Australia, or India.
                    country_risk: LOW, MEDIUM, or HIGH.
                    time_of_day: MORNING, AFTERNOON, EVENING, or NIGHT.
                    cardtype: CREDIT, DEBIT, or PREPAID.
            
            Approach to Apply:
            
            Approach (A): Generative Models
                - Emulate learned distributions and dependencies.
                - Generate 3-5 high-quality samples per user_id (Balanced classes if possible).
                - Reason about fault_predicted based on patterns.

            Strict Output Format:
                1. Provide a single TABLE for Approach A containing ALL users.
                3. Provide the detailed explanation and accuracy comparison vs XGBOOST after the tables.
                
            IMPORTANT: Ensure every user_id from the input file appears in the tables below.
            
            Assistant:"""
            
        # 5. Invoke LLM 
        updated_content = get_llm_response(prompt_template)
        csv_content = process_to_csv(updated_content)

        csv_file_key = f"{output_folder}csv_synthetic_data.csv" 
        
        # Save to S3
        s3.put_object(
            Bucket=bucket_name,
            Key=csv_file_key,
            Body=csv_content.encode('utf-8'),
            ContentType='text/csv' # Updated ContentType
        )

        # 6. Save to S3
        updated_file_key = f"{output_folder}synthetic_data_updated.txt" 
        # output_file_s3_key = f"{bucket_name}{updated_file_key}" # Note: Ensure slash handling if needed in your bucket logic
 
        s3.put_object(
            Bucket=bucket_name,
            Key=updated_file_key,
            Body=updated_content.encode('utf-8'),
            ContentType='text/plain' 
        )
        
        s3_path = f"s3://{bucket_name}/{updated_file_key}"
        logger.info(f"Updated content saved to: {s3_path}")
        
        # 7. Return success response
        final_message = (
            f"TOOL_STATUS: SUCCESS. Synthetic data generated for ALL users.\n"
            f"**UPDATED_POLICY_PATH: {s3_path}**\n\n"
            f"Content Preview:\n---\n{updated_content}"
        )

        final_output = build_agent_response(action_group, function_name, final_message, response_state="REPROMPT")
        return final_output

    except Exception as e:
        logger.error(f"Failed to process request: {e}")
        error_message = f"An internal error occurred: {str(e)}"
        return build_agent_response(action_group, function_name, error_message, response_state="FAILURE")