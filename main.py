import streamlit as st
import json
import pandas as pd
from time import sleep
import re
from pathlib import Path
import glob
from dataclasses import dataclass
from typing import List
from litellm import (
    completion,
    RateLimitError,
    ContextWindowExceededError,
    BadRequestError,
    AuthenticationError,
    APIConnectionError,
    APIError
)
import os

@dataclass
class Template:
    name: str
    prompt: str
    columns: List[str]

def description_to_json(description, system_prompt, columns):
    temperature = 0
    max_retries = 5
    retry_count = 0
    base_delay = 1  # Base delay in seconds

    while retry_count < max_retries:
        try:
            response = completion(
                model=st.session_state.selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Description: {description}"}
                ],
                temperature=temperature,
                max_tokens=1024,
                drop_params=True,
            )

            output = json.loads(response.choices[0].message.content)
            # Validate output structure
            for col_name in columns:
                output[col_name]['count']
            return output

        except RateLimitError as e:
            # Exponential backoff for rate limits
            delay = base_delay * (2 ** retry_count)
            sleep(delay)
            retry_count += 1
            continue

        except ContextWindowExceededError as e:
            # This is a permanent error - the input is too long
            raise Exception("Input text is too long for the model's context window") from e

        except BadRequestError as e:
            # Try increasing temperature for malformed responses
            if temperature < 0.9:
                temperature += 0.1
                continue
            raise Exception(f"Invalid request: {str(e)}") from e

        except AuthenticationError as e:
            # Authentication errors should fail immediately
            raise Exception("Invalid API key or authentication error") from e

        except APIConnectionError as e:
            # Network errors can be retried
            sleep(base_delay)
            retry_count += 1
            continue

        except APIError as e:
            # Generic API errors - retry with backoff
            delay = base_delay * (2 ** retry_count)
            sleep(delay)
            retry_count += 1
            continue

        except json.JSONDecodeError as e:
            # Try increasing temperature for malformed JSON
            if temperature < 1.0:
                temperature += 0.1
                continue
            raise Exception("Failed to parse model response as JSON") from e

        except Exception as e:
            # Unexpected errors
            retry_count += 1
            if retry_count >= max_retries:
                raise Exception(f"Max retries reached. Last error: {str(e)}") from e

    raise Exception("Maximum retry attempts exceeded")

def process_data(df, system_prompt, columns, job_id, status_text):
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create a copy of the dataframe to avoid displaying intermediate results
    result_df = df.copy()
    
    with st.empty():  # Create a container for status updates
        for index, row in df.iterrows():
            try:
                description = row['Description']
                output = description_to_json(description, system_prompt, columns)
                
                for col_name in columns:
                    result_df.loc[index, col_name] = output[col_name]['count']
                
                status_text.text(f"Processing row {index + 1} of {len(df)}")
                
            except Exception as e:
                error_msg = str(e)
                status_text.error(f"Error in row {index + 1}: {error_msg}")
                raise Exception(f"Processing stopped due to error in row {index + 1}: {error_msg}")
    
    return result_df

def test_prompt(df, system_prompt, columns):
    """Test the prompt with a random row from the dataset"""
    # Clear previous test results if they exist
    if 'test_container' in st.session_state:
        st.session_state.test_container.empty()
    
    # Create a new container for test results
    st.session_state.test_container = st.container()
    
    with st.session_state.test_container:
        try:
            random_row = df.sample(n=1).iloc[0].astype(str)
            st.write("Testing with random row:")
            
            # Create DataFrame and reorder columns to put Description first
            test_df = pd.DataFrame([random_row.to_dict()])
            cols = ['Description'] + [col for col in test_df.columns if col != 'Description']
            test_df = test_df[cols]
            
            st.dataframe(test_df)
            
            if 'Description' not in random_row:
                st.error("CSV file must contain a 'Description' column")
                return
                
            output = description_to_json(random_row['Description'], system_prompt, columns)
            st.write("Output:")
            st.json(output)
        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty")
        except Exception as e:
            st.error(f"Error processing test: {str(e)}")

def load_templates():
    """Load all templates from the templates directory"""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    templates = {}
    template_files = glob.glob("templates/*.json")
    for f in template_files:
        with open(f) as file:
            data = json.load(file)
            templates[Path(f).stem] = Template(**data)
    
    return templates

def save_template(template: Template):
    """Save a template to the templates directory"""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    template_data = {
        "name": template.name,
        "prompt": template.prompt,
        "columns": template.columns
    }
    
    with open(templates_dir / f"{template.name}.json", "w") as f:
        json.dump(template_data, f, indent=2)

def initialize_session_state():
    if "api_key" not in st.session_state:
        st.session_state.api_key = None
    if "job_id" not in st.session_state:
        st.session_state.job_id = None
    if "processed_df" not in st.session_state:
        st.session_state.processed_df = None
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gpt-4"
    if "selected_provider" not in st.session_state:
        st.session_state.selected_provider = "OpenAI"

def main():
    # Add custom CSS at the start of main()
    st.markdown("""
        <style>
        /* Custom green theme */
        .stButton > button {
            background-color: #2e7d32;
            color: white;
        }
        .stButton > button:hover {
            background-color: #1b5e20;
        }
        /* Secondary button style */
        .stButton > button[data-baseweb="button"][kind="secondary"] {
            background-color: #81c784;
            color: black;
        }
        .stButton > button[data-baseweb="button"][kind="secondary"]:hover {
            background-color: #66bb6a;
        }
        </style>
    """, unsafe_allow_html=True)

    initialize_session_state()
    
    st.title("SF Permit Data Parser")
    
    # Overview
    st.markdown("""
        🔍 This tool helps analyze plumbing permit descriptions to identify and count specific types of equipment installations.
    """)

    # How to use
    with st.expander("📖 How to Use", expanded=False):
        st.markdown("""
            1. **API Key Setup:**
               - For OpenAI: Visit [OpenAI API Keys](https://platform.openai.com/api-keys) to generate a key
               - For Anthropic: Go to [Anthropic Console](https://console.anthropic.com/account/keys) to create a key
               - For Groq: Access [Groq Console](https://console.groq.com/keys) to get your API key
            2. Upload a CSV file containing permit descriptions (must have a column named 'Description')
            3. Select a template for the type of equipment you want to analyze
            4. Download the parsed results and equipment counts when processing is complete
            
            **About the Buttons:**
            - **Test Prompt**: Tests the template on a single sample description to verify the prompt works as expected before processing the full dataset
            - **Start Processing**: Begins analyzing all descriptions in your uploaded CSV file using the selected template and model
        """)

    # Important note
    st.markdown("""
        ⚠️ **Important Note:** Processing large files can take a significant amount of time. Make sure to keep your computer awake and browser window open during processing to avoid interruptions. If the process stops, you will need to start over from the beginning.
    """)
    # Create two columns for model selection and API key
    col1, col2 = st.columns([1, 1])
    
    # Model selection in left column
    with col1:
        providers = ["OpenAI", "Anthropic", "Groq"]
        st.session_state.selected_provider = st.selectbox(
            "Select Provider",
            providers,
            index=providers.index("OpenAI")
        )

        # Model options based on provider
        if st.session_state.selected_provider == "OpenAI":
            models = [
                "gpt-4o-mini",
            ]
        elif st.session_state.selected_provider == "Anthropic":
            models = [
                "claude-3-5-sonnet-20240620",
            ]
        else:  # Groq
            models = [
                "groq/llama-3.1-70b-versatile",
            ]
        
        st.session_state.selected_model = st.selectbox(
            "Select Model",
            models,
            index=0,
            help="💡 Select your preferred model"
        )
    
    # API key input in right column
    with col2:
        api_key = st.text_input(
            f"Enter your {st.session_state.selected_provider} API Key " +
            f"([Get API Key]({'https://platform.openai.com/api-keys' if st.session_state.selected_provider == 'OpenAI' else 'https://console.anthropic.com/account/keys' if st.session_state.selected_provider == 'Anthropic' else 'https://console.groq.com/keys'}))",
            type="password",
            help=f"💡 To get your API key: Visit the {st.session_state.selected_provider} website and generate a new API key"
        )
        if api_key:
            # Display cost warning based on provider
            if st.session_state.selected_provider in ["OpenAI", "Anthropic"]:
                st.warning(f"⚠️ {st.session_state.selected_provider} API usage will incur charges based on your usage. Make sure you understand their pricing model.")
            elif st.session_state.selected_provider == "Groq":
                st.success("ℹ️ Groq API is currently free to use!")
                
            st.session_state.api_key = api_key
            # Set the appropriate environment variable based on provider
            if st.session_state.selected_provider == "OpenAI":
                os.environ["OPENAI_API_KEY"] = api_key
            elif st.session_state.selected_provider == "Anthropic":
                os.environ["ANTHROPIC_API_KEY"] = api_key
            else:  # Groq
                os.environ["GROQ_API_KEY"] = api_key
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file and st.session_state.api_key:
        # Read and display CSV
        df = pd.read_csv(uploaded_file)
        
        # Template selection/creation
        templates = load_templates()
        col1, col2 = st.columns([2, 1])
        
        with col1:
            template_options = list(templates.keys()) + ["Create new"]
            selected_template = st.selectbox(
                "Select a template",
                template_options
            )
        
        if selected_template == "Create new":
            # Clear existing columns when creating new template
            if 'columns' not in st.session_state or selected_template != st.session_state.get('last_template'):
                st.session_state.columns = []
                st.session_state.last_template = selected_template
                
            template_name = st.text_input("Enter template name")
            system_prompt = st.text_area("Enter the prompt content")
            
            # Column management for new template
            st.subheader("Template Columns")
            col1, col2 = st.columns([6, 1])
            with col1:
                new_column = st.text_input("New column name")
            with col2:
                st.markdown('<div style="margin: 28px;"></div>', unsafe_allow_html=True)
                if st.button("➕ Add", type="primary", use_container_width=True):
                    if new_column and new_column not in st.session_state.columns:
                        st.session_state.columns.append(new_column)
                        st.rerun()
            
            # Display current columns
            if st.session_state.columns:
                st.write("Current columns:")
                for i, col in enumerate(st.session_state.columns):
                    cols = st.columns([6, 1])
                    with cols[0]:
                        st.markdown(f"- **{col}**")
                    with cols[1]:
                        # Wrap button in container
                        button_container = st.container()
                        if button_container.button("🗑️", key=f"delete_column_{i}", use_container_width=True):
                            st.session_state.columns.remove(col)
                            st.rerun()
            
            if st.button("Save Template", type="primary"):
                if template_name and system_prompt and st.session_state.columns:
                    if template_name in ["electric", "gas"]:
                        st.error("Cannot overwrite protected templates.")
                    else:
                        new_template = Template(
                            name=template_name,
                            prompt=system_prompt,
                            columns=st.session_state.columns
                        )
                        save_template(new_template)
                        st.success("Template saved!")
                        st.rerun()
        else:
            template = templates[selected_template]
            st.session_state.columns = template.columns
            
            col1, col2 = st.columns([6, 1])
            with col1:
                st.text_area("Template prompt", value=template.prompt, height=400, disabled=True)
            with col2:
                if selected_template not in ["electric", "gas"]:
                    st.write("")
                    st.write("")
                    # Wrap button in container
                    button_container = st.container()
                    if button_container.button("🗑️", key="delete_template", use_container_width=True):
                        (Path("templates") / f"{selected_template}.json").unlink()
                        st.success(f"Template '{selected_template}' deleted!")
                        st.rerun()
        
        if selected_template != "Create new":
            template = templates[selected_template]
            system_prompt = template.prompt
            
            # Add JSON format instruction to system prompt
            json_format = {col: {"reasoning": "string", "count": 0} for col in template.columns}
            full_prompt = system_prompt + f"\nOutput the results in JSON format exactly like this structure:\n{json.dumps(json_format, indent=2)}"
            
            if st.button("🧪 Test Prompt", type="secondary"):
                test_prompt(df, full_prompt, template.columns)
            
            if st.button("▶️ Start Processing", type="primary"):
                status_text = st.empty()
                processed_df = process_data(
                    df, 
                    full_prompt, 
                    template.columns,
                    st.session_state.job_id,
                    status_text
                )
                
                st.session_state.processed_df = processed_df
                
                st.download_button(
                    label="📥 Download processed data",
                    data=processed_df.to_csv(index=False).encode('utf-8'),
                    file_name="processed_data.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main() 