import boto3
import json
import logging
import textwrap
from botocore.config import Config
import io
import csv
import json
import math
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)
from datetime import datetime

class DateTimeEncoder(json.JSONEncoder):
    """A custom JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(DateTimeEncoder, self).default(obj)

# --- AWS Configuration ---
REGION = 'us-west-2' 

# --- AWS Clients ---
s3 = boto3.client('s3')

# Custom config for Bedrock to handle longer running requests
custom_config = Config(
    read_timeout=1000, 
    connect_timeout=1500,
    retries={'max_attempts': 0}
)
bedrock_runtime = boto3.client('bedrock-runtime', region_name=REGION, config=custom_config)



# --- CRITICAL HELPER FUNCTION ---

def build_agent_response(action_group, function_name, message, response_state):
    """
    Helper function to build the Bedrock Agent compliant response JSON.
    Uses the correct REPROMPT/FAILURE responseState and only the TEXT content type.
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

# --- Other Helper Functions (Omitted for brevity, assume correct) ---


def get_llm_response(prompt: str) -> str:
    # ... (Your Bedrock invocation logic remains the same)
    model_id = 'arn:aws:bedrock:us-west-2:640237069320:inference-profile/global.anthropic.claude-sonnet-4-5-20250929-v1:0'
    # model_id= 'anthropic.claude-3-5-sonnet-20240620-v1:0'
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 30000,
        "temperature": 0.0,
        "anthropic_version": "bedrock-2023-05-31"
    })
    response = bedrock_runtime.invoke_model(
        modelId=model_id, body=body, contentType='application/json', accept='application/json'
    )
    response_body = json.loads(response['body'].read())
    return response_body.get('content', [{}])[0].get('text', 'LLM did not return text.')

def get_s3_file_content(bucket: str, key: str) -> str:
    # ... (Your S3 content retrieval logic remains the same)
    response = s3.get_object(Bucket=bucket, Key=key)
    return response['Body'].read().decode('utf-8')

def save_markdown_table_as_csv(bucket: str, key: str, markdown_text: str):
    """
    Extracts a Markdown table from text and saves it as a CSV file to S3.
    """
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    
    lines = markdown_text.split('\n')
    in_table = False
    
    for line in lines:
        # Detect table start (looking for the header row with user_id)
        if '|' in line and 'user_id' in line.lower():
            in_table = True
        
        if in_table:
            # Stop if we hit an empty line or a line without pipes (end of table)
            if not line.strip() or '|' not in line:
                if len(line.strip()) > 0 and '---' not in line:
                    in_table = False
                continue
            
            # Extract columns by splitting on pipe and stripping whitespace
            columns = [col.strip() for col in line.split('|') if col.strip()]
            
            # Skip the Markdown separator line (e.g., |---|---|)
            if all(c.strip('-') == '' for c in columns):
                continue
            
            if columns:
                writer.writerow(columns)

    # Only upload if data was actually found
    if csv_buffer.tell() > 0:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=csv_buffer.getvalue().encode('utf-8'),
            ContentType='text/csv'
        )
        return True
    return False


# --- Main Lambda Handler ---

def lambda_handler(event, context):
   
    logger.info("Event received: " + json.dumps(event))

    action_group = event.get('actionGroup')
    function_name = event.get('function')
    
    # --- 1. Agent Input Retrieval ---
    try:
        if not action_group or not function_name:
            raise KeyError("Missing actionGroup or function name in event.")

        params = {p['name']: p['value'] for p in event.get('parameters', [])}
        
        if function_name != 'fraud_detection_group_function':
             raise ValueError(f"Unknown function: {function_name}")
             
        bucket_name = params['bucket_name']
        key_new_document = params['new_document_key'] 
        # summary_key = params['summary_key']
        # print(summary_key)
        customer_data = params['customer_data']
        merchant_data= params['merchant_data']
        bank_train_data= params['bank_train_data']
       
        output_folder = params.get('output_folder', 'updated_output/') 
       
    except (KeyError, ValueError, TypeError) as e:
        error_message = f'Missing or invalid Agent parameter. Details: {str(e)}'
        return build_agent_response(action_group, function_name, error_message, response_state="REPROMPT")

    try:
        logger.info(f"Bucket: {bucket_name}, New Doc Key: {key_new_document}, ")

        file_content = get_s3_file_content(bucket_name, key_new_document)
        customer_file=get_s3_file_content(bucket_name, customer_data)
        merchant_file=get_s3_file_content(bucket_name, merchant_data)
        bank_train_file=get_s3_file_content(bucket_name, bank_train_data)
        
        logger.info(f"Successfully read New Document content.")
        updated_file_key_json = f"updated_output/_bank_model_performance_agent.json"
        output_fp = f"{bucket_name}fp.csv"

        #construct from knowledgebase
      
      
        # 4. Construct the prompt 
        # ❌ CRITICAL PROMPT UPDATE: ADD INSTRUCTIONS FOR TABLES
        # 4. Find the legitimate and Non-Legitimate transaction in train data set and customer file {customer_file} and {bank_train_file}
        prompt_template = f"""
Human: Model Performance Agent Analysis:

Objective:
1. Analyze XGBOOST Model features and performance (specifically the 15 False Positives and 99 False Negatives).
2. Categorize all 114 transactions as Legitimate/Non-Legitimate using behavioral profiling.
3. Perform Merchant and Customer profile risk matching.

### 1. XGBOOST Model Analysis:
- Explain features and provide a summary based on {file_content}.
- Analyze why the model failed to detect 99 fraudulent transactions (False Negatives) and incorrectly flagged 15 (False Positives).

### 2. Transaction Analysis & Behavioral Justification
- Process ALL 114 transactions from {file_content}.
- **Categorization Rules**:
    - **False Positive (FP)**: Predicted Fraud (1) but Actual Legitimate (0).
    - **False Negative (FN)**: Predicted Legitimate (0) but Actual Fraud (1).
    - Categorize each row in a new column **"Transaction_Status"** as 'Legitimate' or 'Non-Legitimate' based on these rules:
            Legitimate: fraud_score < 0.85, normal velocity, and matching country profile.
            Non-Legitimate: fraud_score >= 0.85, velocity spikes, or unusual merchant category.
- **Output Table Format**: 
  Include: user_id, transaction_id, amount, fraud_score, is_fraud_predicted, is_fraud_actual, Is_False_Positive, Is_False_Negative, Transaction_Status, and **Reason**.

- **Reason Column Requirements (CRITICAL)**:
  For every row, provide a concise explanation in this exact narrative format:
  "User (Name) has a tenure of <customer_tenure_months> months and card age of <card_age> months. Based on a <low/medium/high> inferred income, the transaction of <amount> at <merchant_category> via <channel> [aligns/does not align] with typical patterns. Justification: [Analyze KYC flags, preferred channels, spending spikes, merchant risk, geolocation, and time of day]. Specifically, [explain why this is a False Positive, False Negative, or Correct hit based on the fraud_score vs actual behavior]."

### 3. Customer & Merchant Risk Matching
- Match {file_content} user_ids with {customer_file} for KYC_STATUS and Tenure.
- Use {merchant_file} to classify Merchant Risk and analyze if geolocations/channels are unusual.

---
### Final Output Format:
- **Section 1**: XGBOOST Summary (Analysis of the 114 data points).
- **Section 2**: Comprehensive Table of ALL 114 transactions. (Ensure the 'Reason' column follows the behavioral narrative requested above).
- **Section 3**: Error Logic Breakdown (Deep dive into the 15 FPs and 99 FNs).
- **Section 4**: Customer Profile & KYC Status matching table.
- **Section 5**: Merchant Risk & Geolocation Analysis summary.

Rules:
- Use Markdown tables for all sections.
- Maintain temperature=0 logic.
- Ensure the 'Reason' column remains concise but covers all requested behavioral variables (Income, Tenure, KYC, Merchant Risk).

Assistant:
"""


### 4. Customer Profile Matching 

# - For each user_id in {file_content} Matches with {customer_file}.
#     - If matched:        
# Rules (customizable):
# Each Case Define Separately below rules get the detailed explanation of all parameters why legitimate or non-legitimate from{customer_file} user_id matches {file_content}.
# 1) Legitimate transaction criteria (all must hold unless overridden):
#    - fraud_score < 0.59999
#    - velocity_1h and velocity_24h within normal bands for the user (e.g., <= 3 txns in 1h, <= 20 txns in 24h).
#    - If user is Domestic: transaction country == domestic_country.
#    - Amount within ±3× interquartile range around user's historical median (fallback: within 2× avg_txn_amount).
#    - Give Detailed Explanation of this user
# 2) Non-Legitimate transaction indicators (any one):
#    - fraud_score >= 0.6
#    - Sudden velocity spike beyond thresholds.
#    - Unusual merchant_category for this user (not seen in last 90 days).
#    - International transaction for a Domestic user; or high-risk corridor for an International user.
#    - user_id not found in customer_file.
#    -Give Detailed Explanation of this user
# 3) Domestic vs International user: If international_txn_ratio >= 0.5, mark user as International; else Domestic.
# Based on above rule category give detail explanation for all user_id  Based on 3 criteria 1. Fraud score,2.Sudden velocity spike beyond thresholds,3. Unusual merchant_category for this user (not seen in last 90 days) 4.transacation mismatch eg: transaction done on international if he is domestic user, in {file_content} matches user_id{customer_file},  classify which are all legitimate transaction and non legitimate transactions with detailed explanation.

# -**Section 5:** Legitimate and Non-Legitimate transactions based on transactions, fraud_score,  from {bank_train_file} with detailed explanation .

        

        # 5. Invoke LLM and get the updated content
        updated_content = get_llm_response(prompt_template)

        #5.1 Save the File in csv to s3 response
        



        #stop
       
        # 6. Determine the new S3 key and save the output
       
        updated_file_key = f"updated_output/_bank_model_performance_agent.txt"
        updated_file_key_json = f"updated_output/_bank_model_performance_agent.json"
        updated_file_key_csv = "credit-card-fdupdated_output/_bank_model_performance_agent.csv"
        output_file_s3_key = f"{bucket_name}{updated_file_key}"
        output_file_s3_key_json = f"{bucket_name}{updated_file_key_json}"
        
        csv_success = save_markdown_table_as_csv(
            bucket_name, 
            updated_file_key_csv, 
            updated_content
        )
        
        s3.put_object(
            Bucket=bucket_name,
            Key=output_file_s3_key,
            Body=updated_content.encode('utf-8'),
            ContentType='application/text' 
        )

        s3.put_object(
            Bucket=bucket_name,
            Key=output_file_s3_key_json,
            Body=updated_content.encode('utf-8'),
            ContentType='application/json' 
        )
        #  ContentType='application/json' 
       
        s3_path = f"s3://{bucket_name}/{updated_file_key}"
        
        logger.info(f"Updated policy saved to: {s3_path}")
        
        # 7. Return success response (Agent Format)
        final_message = (
            f"TOOL\_STATUS: SUCCESS. The policy has been updated by the LLM and saved to S3. "
            f"The key for the updated policy is needed for any next steps. "
            f"**UPDATED\_POLICY\_PATH: {s3_path}**\n\n"
            f"The final content is:\n---\n{updated_content}\n"
            
        )
           
        return build_agent_response(action_group, function_name, final_message, response_state="REPROMPT")

    except Exception as e:
        logger.error(f"Failed to process request: {e}")
        error_message = f"An internal error occurred during policy update: {str(e)}"
        return build_agent_response(action_group, function_name, error_message, response_state="FAILURE")