from graphs import email_assistant_workflow
from pprint import pprint

# Define the test state with the necessary structure
state = {
    "email_input": {
        "From": "sender@example.com",
        "To": "receiver@example.com",
        "Subject": "Meeting Request",
        "Body": "Can we meet tomorrow?"
    },
    "email_type": "",  # You can adjust this depending on your use case
    "query": "reply of this message",  # The query for generating a reply
    "previous_email": [],  # Empty list as no prior emails are provided
    "user_id": "user_12345",  # Unique user ID
    "generation": {},  # Empty, can be populated later with generated content
    "sub_router_state": "",  # No sub-router state for now
    "improvement_list": [],  # No improvements required at this point
    "needing_improvement": "",  # Nothing needing improvement for now
    "classification": ""  # Classification can be added later if required
}

# Assuming you have multiple test cases, you can define them like so
test_cases = [state]  # Add more test cases as needed

# Stream the results
def stream_workflow(state):
    # Assuming `app.stream()` is the method for streaming results
    for result in main_agent.stream(state):  # main_agent.stream() to simulate streaming
        yield result

# Compile the main agent workflow
main_agent = email_assistant_workflow()

# Loop over test cases
for i, test_state in enumerate(test_cases, 1):
    print(f"\n=== Test Case {i} ===\n")
    for output in stream_workflow(test_state):
        pprint(output)

# Invoke the main agent with the given state
response = main_agent.invoke(state)

# Output the response
print("\nFinal Response:")
pprint(response)
