import streamlit as st
import random
import pandas as pd
import os
import time
import plotly.express as px
from langchain.tools import BaseTool, StructuredTool, tool, Tool
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
import json
from statistics import mode


# Set page config for better appearance
st.set_page_config(page_title="BS-CUSTOMER", page_icon="💬", layout="wide")

# # Configure logging
# logging.basicConfig(filename='app.log', level=logging.ERROR)

# Custom CSS for styling
st.markdown("""
    <style>
    /* Set the main background color */
    .css-18e3th9 {
        background-color: #ffeb3b !important;
        color: black;
    }
    /* Set the sidebar background color */
    .css-1d391kg {
        background-color: #ff9800 !important;
        color: white;
    }
    /* Set the button colors */
    .stButton>button {
        background-color: #f44336;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #d32f2f;
        color: white;
    }
    /* Set the selectbox colors */
    .stSelectbox>div>div>button {
        background-color: #ff6f00;
        color: white;
    }
    .stSelectbox>div>div>button:hover {
        background-color: #e65100;
        color: white;
    }
    /* Set the checkbox colors */
    .stCheckbox>div {
        color: white;
    }
    /* Set the chat message colors */
    .stChatMessage {
        background-color: #333333;
        color: white;
    }
    .stChatMessage>div>div>div {
        color: white;
    }
    /* Set the text input colors */
    .stTextInput>div>div>input {
        background-color: #444444;
        color: white;
    }
    .stTextInput>div>div>button {
        background-color: #f44336;
        color: white;
        border: none;
    }
    .stTextInput>div>div>button:hover {
        background-color: #d32f2f;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if 'chat' not in st.session_state:
        st.session_state['chat'] = [{"content": "Hi, I need your help", "role": "ai"}]
    if 'chat-history' not in st.session_state:
        st.session_state['chat-history'] = [{"content": "Hi, I need your help", "role": "ai"}]
    if 'selected_option' not in st.session_state:
        st.session_state['selected_option'] = None
    if 'last_played_index' not in st.session_state:
        st.session_state['last_played_index'] = -1
    if 'page' not in st.session_state:
        st.session_state['page'] = 'home'
    if 'current_caller_id' not in st.session_state:
        st.session_state['current_caller_id'] = None
    if 'current_question_index' not in st.session_state:
        st.session_state['current_question_index'] = 0
    if 'responses' not in st.session_state:
        st.session_state['responses'] = []
    if 'question_start_time' not in st.session_state:
        st.session_state['question_start_time'] = None
    

initialize_session_state()

# Sidebar navigation
with st.sidebar:

    if st.button("Home"):
        st.session_state['page'] = 'home'
    if st.button("QnAPlayer"):
        st.session_state['page'] = 'QnAPlayer'
    if st.button("QnAPlayer+"):
        st.session_state['page'] = 'QnAPlayer+'
    # if st.button("Simulator"):
    #     st.session_state['page'] = 'Simulator'


# Load questions
def load_questions():
    try:
        return pd.read_csv('Care_questions_no_repeats.csv')
    except FileNotFoundError:
        st.error("Questions dataset not found!")
        return pd.DataFrame(columns=['caller_id', 'question', 'parentintent', 'childintent', 'answer'])

# Save user responses
def save_user_response(domain, sub_domain, question, user_answer, actual_answer, caller_id, response_time):
    response_data = {
        'domain': domain,
        'sub_domain': sub_domain,
        'question': question,
        'user_answer': user_answer,
        'actual_answer': actual_answer,
        'caller_id': caller_id,
        'response_time': response_time
    }
    df = pd.DataFrame([response_data])
    df.to_csv('V1_responses.csv', mode='a', header=not os.path.exists('user_responses.csv'), index=False)



def display_about():
    st.header("Welcome to BS-CUSTOMER")
    
    st.markdown("""
        BS-CUSTOMER is a simulator designed to train BS Customer Support agents using three main services: QnAPlayer, QnAPlayer+, and Simulator.
        - **QnAPlayer**: Tests the response of the BS agent.
        - **QnAPlayer+**: Evaluates the relevancy of the BS agent's answers using similarity scores.
        
        
        Our goal is to provide robust training to BS agents, ensuring they can handle various scenarios effectively.
    """)

    st.subheader("Features")
    st.markdown("""
        - **Service Management**: Get help with managing your services efficiently.
        - **Billing**: Clear your doubts regarding billing and payments.
        - **Account Management**: Manage your account settings and preferences with ease.
    """)

    st.subheader("How to Use")
    st.markdown("""
        To start training with BS-CUSTOMER, select one of the services (QnAPlayer, QnAPlayer+) from the dropdown menu.
    """)

    service_option = st.selectbox("Select Service", ["QnAPlayer", "QnAPlayer+"])

    if service_option == "QnAPlayer":
        st.header("About QnAPlayer Service - Question & Answer")
    
        st.markdown("""
            The QnAPlayer service of BS-CUSTOMER simulates interactions to train BS Customer Support agents using a structured Q&A approach:
            
            **How It Works:**
            1. **Select Domain and Sub-Domain**: Choose from various domains and scenario related to customer queries.
            2. **Start Interaction**: Begin the interaction to receive questions from the selected domain.
            3. **Submit Answer**: Provide answers and receive feedback on correctness.
            4. **Next Interaction**: Move to the next set of questions for further training.
            5. **Get Analysis**: View detailed analysis of your interactions.
            
            This service helps in testing the response capabilities of BS agents across different domains effectively.
        """)
    elif service_option == "QnAPlayer+":
        st.header("About QnAPlayer+ Service - Question & Answer with Similarity Score")
    
        st.markdown("""
            The QnAPlayer+ service of BS-CUSTOMER enhances the training process by evaluating answers using similarity scores:
            
            **Key Features:**
            1. **Domain and Sub-Domain Selection**: Choose specific domains and sub-domains for focused training.
            2. **Interactive Q&A**: Engage in Q&A sessions where answers are evaluated based on similarity to correct responses.
            3. **Real-time Feedback**: Receive instant feedback on the accuracy of your responses.
            4. **Detailed Analysis**: Get insights into response times and similarity scores for each interaction.
            
            This service ensures that BS agents provide relevant and accurate information aligned with expected responses.
        """)


def display_v1():
    st.title("QnAPlayer")

    questions_df = load_questions()

    with st.sidebar:
        st.header("Select a Domain and Scenario")
        selected_domain = st.selectbox("Select a Domain", questions_df['parentintent'].unique(), key='domain_v1')
        selected_sub_domain = st.selectbox("Select a Scenario", questions_df[questions_df['parentintent'] == selected_domain]['childintent'].unique(), key='sub_domain_v1')

        if st.session_state.get('selected_option_v1') != (selected_domain, selected_sub_domain):
            st.session_state['selected_option_v1'] = (selected_domain, selected_sub_domain)
            st.session_state.pop('current_caller_id_v1', None)
            st.session_state.pop('current_question_index_v1', None)
            st.session_state.pop('responses_v1', None)
            st.session_state.pop('chat_v1', None)
            st.experimental_rerun()

    if st.session_state.get('current_caller_id_v1') is None:
        st.write("Please select a domain and scenario, then click 'Start Interaction'.")

        if st.button("Start Interaction"):
            filtered_df = questions_df[(questions_df['parentintent'] == selected_domain) & (questions_df['childintent'] == selected_sub_domain)]
            if not filtered_df.empty:
                random_caller_id = random.choice(filtered_df['caller_id'].unique())
                st.session_state['current_caller_id_v1'] = random_caller_id
                st.session_state['current_question_index_v1'] = 0
                st.session_state['responses_v1'] = []
                st.session_state['chat_v1'] = []
                st.session_state['question_start_time_v1'] = None
                st.session_state['user_answer_v1'] = ''
                st.experimental_rerun()
            else:
                st.write("No questions available for this domain and scenario.")
                st.session_state.pop('current_caller_id_v1', None)
    else:
        filtered_df = questions_df[questions_df['caller_id'] == st.session_state['current_caller_id_v1']]
        if st.session_state['current_question_index_v1'] < len(filtered_df):
            current_question = filtered_df.iloc[st.session_state['current_question_index_v1']]
            if st.session_state['question_start_time_v1'] is None:
                st.session_state['question_start_time_v1'] = time.time()
                st.session_state['chat_v1'].append({"content": f"Question {st.session_state['current_question_index_v1'] + 1}: {current_question['question']}", "role": "ai"})

            if len(st.session_state['chat_v1']) > 0:
                for message in st.session_state['chat_v1']:
                    st.chat_message(message['role']).write(message['content'])

            user_answer_key = f"user_answer_v1_{st.session_state['current_question_index_v1']}"

            user_answer = st.text_area("Your Answer", key=user_answer_key)

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Submit Answer"):
                    response_time = time.time() - st.session_state['question_start_time_v1']
                    st.session_state['question_start_time_v1'] = None


                    save_user_response(
                        domain=current_question['parentintent'],
                        sub_domain=current_question['childintent'],
                        question=current_question['question'],
                        user_answer=user_answer,
                        actual_answer=current_question['answer'],
                        caller_id=st.session_state['current_caller_id_v1'],
                        response_time=response_time
                    )

                    st.session_state['responses_v1'].append({
                        'question': current_question['question'],
                        'user_answer': user_answer,
                        'actual_answer': current_question['answer'],
                        'response_time': response_time
                    })
                    st.session_state['chat_v1'].append({"content": user_answer, "role": "user"})
                    st.session_state['current_question_index_v1'] += 1
                    st.experimental_rerun()

            with col2:
                if st.button("Stop Interaction"):
                    st.session_state.pop('current_caller_id_v1', None)
                    st.session_state.pop('current_question_index_v1', None)
                    st.session_state.pop('responses_v1', None)
                    st.session_state.pop('chat_v1', None)
                    st.session_state.pop(user_answer_key, None)
                    st.experimental_rerun()

            with col3:
                if st.button("Restart Interaction"):
                    filtered_df = questions_df[(questions_df['parentintent'] == selected_domain) & (questions_df['childintent'] == selected_sub_domain)]
                    if not filtered_df.empty:
                        random_caller_id = random.choice(filtered_df['caller_id'].unique())
                        st.session_state['current_caller_id_v1'] = random_caller_id
                        st.session_state['current_question_index_v1'] = 0
                        st.session_state['responses_v1'] = []
                        st.session_state['chat_v1'] = []
                        st.session_state['question_start_time_v1'] = None
                        st.experimental_rerun()
                    else:
                        st.write("No questions available for this domain and scenario.")
                        st.session_state.pop('current_caller_id_v1', None)

        else:
            st.write("You have answered all the questions for this interaction. Please click 'Get Analysis' for analysis or 'Next Interaction' for a new caller interaction.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Next Interaction"):
                    filtered_df = questions_df[(questions_df['parentintent'] == selected_domain) & (questions_df['childintent'] == selected_sub_domain)]
                    if not filtered_df.empty:
                        random_caller_id = random.choice(filtered_df['caller_id'].unique())
                        st.session_state['current_caller_id_v1'] = random_caller_id
                        st.session_state['current_question_index_v1'] = 0
                        st.session_state['responses_v1'] = []
                        st.session_state['chat_v1'] = []
                        st.experimental_rerun()
                    else:
                        st.write("No questions available for this domain and scenario.")
                        st.session_state.pop('current_caller_id_v1', None)
            
            with col2:
                    if st.button("Get Analysis"):
                        
                        try:
                            st.balloons()
                            responses_df = pd.DataFrame(st.session_state['responses_v1'])
                            st.write("Interaction Details:", responses_df)
                            
                            fig1 = px.bar(
                                responses_df,
                                x='question',
                                y='response_time',
                                title='Response Times for Each Question',
                                labels={'response_time': 'Response Time (s)', 'question': 'Question'},
                            )
                            st.plotly_chart(fig1)


                        except ValueError as e:
                            st.error(f"An error occurred while generating the analysis: {str(e)}")
                            st.write("Please restart the interaction and try again.")
def display_v2():
    st.title("QnAPlayer+")

    questions_df = load_questions()

    with st.sidebar:
        st.header("Select a Domain and Scenario")
        selected_domain = st.selectbox("Select a Domain", questions_df['parentintent'].unique(), key='domain_v2')
        selected_sub_domain = st.selectbox("Select a Scenario", questions_df[questions_df['parentintent'] == selected_domain]['childintent'].unique(), key='sub_domain_v2')

        if st.session_state.get('selected_option_v2') != (selected_domain, selected_sub_domain):
            st.session_state['selected_option_v2'] = (selected_domain, selected_sub_domain)
            st.session_state.pop('current_caller_id_v2', None)
            st.session_state.pop('current_question_index_v2', None)
            st.session_state.pop('responses_v2', None)
            st.session_state.pop('chat_v2', None)
            st.experimental_rerun()

    if st.session_state.get('current_caller_id_v2') is None:
        st.write("Please select a domain and scenario, then click 'Start Interaction'.")

        if st.button("Start Interaction"):
            filtered_df = questions_df[(questions_df['parentintent'] == selected_domain) & (questions_df['childintent'] == selected_sub_domain)]
            if not filtered_df.empty:
                random_caller_id = random.choice(filtered_df['caller_id'].unique())
                st.session_state['current_caller_id_v2'] = random_caller_id
                st.session_state['current_question_index_v2'] = 0
                st.session_state['responses_v2'] = []
                st.session_state['chat_v2'] = []
                st.session_state['question_start_time_v2'] = None
                st.experimental_rerun()
            else:
                st.write("No questions available for this domain and scenario.")
                st.session_state.pop('current_caller_id_v2', None)
    else:
        filtered_df = questions_df[questions_df['caller_id'] == st.session_state['current_caller_id_v2']]
        if st.session_state['current_question_index_v2'] < len(filtered_df):
            current_question = filtered_df.iloc[st.session_state['current_question_index_v2']]
            if st.session_state['question_start_time_v2'] is None:
                st.session_state['question_start_time_v2'] = time.time()
                st.session_state['chat_v2'].append({"content": f"Question {st.session_state['current_question_index_v2'] + 1}: {current_question['question']}", "role": "ai"})

            if len(st.session_state['chat_v2']) > 0:
                for message in st.session_state['chat_v2']:
                    st.chat_message(message['role']).write(message['content'])

            user_answer_key = f"user_answer_v2_{st.session_state['current_question_index_v2']}"

            user_answer = st.text_area("Your Answer", key=user_answer_key)

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Submit Answer"):
                    response_time = time.time() - st.session_state['question_start_time_v2']
                    st.session_state['question_start_time_v2'] = None

                    save_user_response(
                        domain=current_question['parentintent'],
                        sub_domain=current_question['childintent'],
                        question=current_question['question'],
                        user_answer=user_answer,
                        actual_answer=current_question['answer'],
                        caller_id=st.session_state['current_caller_id_v2'],
                        response_time=response_time
                    )

                    st.session_state['responses_v2'].append({
                        'question': current_question['question'],
                        'user_answer': user_answer,
                        'actual_answer': current_question['answer'],
                        'response_time': response_time,
                    })
                    st.session_state['chat_v2'].append({"content": user_answer, "role": "user"})
                    st.session_state['current_question_index_v2'] += 1
                    st.experimental_rerun()

            with col2:
                if st.button("Stop Interaction"):
                    st.session_state.pop('current_caller_id_v2', None)
                    st.session_state.pop('current_question_index_v2', None)
                    st.session_state.pop('responses_v2', None)
                    st.session_state.pop('chat_v2', None)
                    st.session_state.pop(user_answer_key, None)
                    st.experimental_rerun()

            with col3:
                if st.button("Restart Interaction"):
                    filtered_df = questions_df[(questions_df['parentintent'] == selected_domain) & (questions_df['childintent'] == selected_sub_domain)]
                    if not filtered_df.empty:
                        random_caller_id = random.choice(filtered_df['caller_id'].unique())
                        st.session_state['current_caller_id_v2'] = random_caller_id
                        st.session_state['current_question_index_v2'] = 0
                        st.session_state['responses_v2'] = []
                        st.session_state['chat_v2'] = []
                        st.session_state['question_start_time_v2'] = None
                        st.experimental_rerun()
                    else:
                        st.write("No questions available for this domain and scenario.")
                        st.session_state.pop('current_caller_id_v2', None)

        else:
            st.write("You have answered all the questions for this interaction. Please click 'Get Analysis' for analysis or 'Next Interaction' for a new caller interaction.")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Next Interaction"):
                    filtered_df = questions_df[(questions_df['parentintent'] == selected_domain) & (questions_df['childintent'] == selected_sub_domain)]
                    if not filtered_df.empty:
                        random_caller_id = random.choice(filtered_df['caller_id'].unique())
                        st.session_state['current_caller_id_v2'] = random_caller_id
                        st.session_state['current_question_index_v2'] = 0
                        st.session_state['responses_v2'] = []
                        st.session_state['chat_v2'] = []
                        st.experimental_rerun()
                    else:
                        st.write("No questions available for this domain and scenario.")
                        st.session_state.pop('current_caller_id_v2', None)

            with col2:
                if st.button("Get Analysis"):
                    try:
                        from langchain_openai import AzureChatOpenAI
                        import os


                        os.environ["AZURE_OPENAI_API_KEY"] = "c5c79306270c45f7b996cdec1975564e"
                        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://openaipoc-01.openai.azure.com"
                        os.environ["AZURE_OPENAI_API_VERSION"] = "2023-05-15"
                        os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "gpt-35-turbo-16k-new"
                        #print(key)
                        def get_chat_model():
                            model = AzureChatOpenAI(
                            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
                        )
                            return model

                        def v2_evaluate(msg):
                            @tool
                            def get_intraction(input:str)->str:
                                "get the chat intraction"
                                return msg


                            etool=[get_intraction]
                            system = """You are the Customer Support Agent evaluator, your job is to evaluate the performance of the Customer support agent by accessing the tool. You should start evaluating if you input is "SCORE"
                            Instructions: 
                            step 1: you should consider HumanMessage with AIMessage in the tool is the agent message.
                            step 2: you should evaluat the based on Quality_of_Response, Communication_Skills, Empathy_and_Tone, Problem_Solving_Skills, Sentiment_Analysis, Chat_Etiquette.
                            step 3: You should give the rounded score for each metric out of five respectively .
                            your output should be python str dict of scores. dont't add any new lines.
                            """
                            prompt = ChatPromptTemplate.from_messages(
                                [
                                    ("system", system),
                                    ("human", "{question}"),
                                ]
                            )
                            query_analyzer = {"question": RunnablePassthrough()} | prompt | get_chat_model()
                            print(query_analyzer.invoke("SCORE").content)
                            return query_analyzer.invoke("SCORE").content
                        
                        def v2_mode_evaluate():
                            json.dumps(v2_evaluate(msg))

                            results = [json.loads(v2_evaluate(msg)) for _ in range(5)]

                            # Initialize a dictionary to store the lists of values for each key
                            aggregated_results = {key: [] for key in results[0].keys()}

                            # Aggregate the results
                            for result in results:
                                for key, value in result.items():
                                    aggregated_results[key].append(value)

                            # Calculate the mode for each key and store in the final dictionary
                            mode_results = {key: mode(values) for key, values in aggregated_results.items()}
                            print(type(mode_results.values))

                            return mode_results
                        
                        def store_scores_to_csv_v2(ev_data):
                            # Calculate the overall score and round it to the nearest integer
                            overall_score = round(sum(ev_data.values()) / len(ev_data))
                            
                            # Add the overall score to the data
                            ev_data['Overall_Score'] = overall_score
                            
                            # Convert the data to a DataFrame
                            df = pd.DataFrame([ev_data])
                            
                            # Write the data to a CSV file, appending if it already exists
                            df.to_csv("V2_Evaluated_Output.csv", mode='a', index=False, header=not pd.io.common.file_exists("V2_Evaluated_Output.csv"))
                            
                            print(f"Data successfully written to Evaluated_Output.csv")


                    



                        st.balloons()
                        responses_df = pd.DataFrame(st.session_state['responses_v2'])

                        # Convert chat messages to the required format
                        chat_interaction = []
                        for msg in st.session_state['chat_v2']:
                            if msg['role'] == 'user':
                                chat_interaction.append(HumanMessage(content=msg['content']))
                            else:
                                chat_interaction.append(AIMessage(content=msg['content']))

                        # Evaluate the interaction
                        v2_evaluate(chat_interaction)


                        score=v2_mode_evaluate()

                        # Store the evaluation data in a CSV file
                        store_scores_to_csv_v2(score)
                        print(score)
                        # Display the overall score
                        score_data=pd.read_csv("V2_Evaluated_Output.csv")
                        last_value=score_data.iloc[-1]['Overall_Score']
         
                        st.markdown(f"<h1 style='text-align: center; color: green;'>Your Score: {last_value}/5</h1>", unsafe_allow_html=True)
                    except ValueError as e:
                        st.error(f"An error occurred while generating the analysis: {str(e)}")
                        st.write("Please restart the interaction and try again.")

class HumanMessage:
    def __init__(self, content):
        self.content = content

class AIMessage:
    def __init__(self, content):
        self.content = content
# Handle navigation
if st.session_state['page'] == 'home':
    display_about()
elif st.session_state['page'] == 'QnAPlayer':
    display_v1()
elif st.session_state['page'] == 'QnAPlayer+':
    display_v2()
# elif st.session_state['page'] == 'Simulator':
#     display_v3()
else:
    display_about()
