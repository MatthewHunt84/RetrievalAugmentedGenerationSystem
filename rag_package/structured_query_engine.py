from llama_index.core import VectorStoreIndex
from llama_index.program.openai import OpenAIPydanticProgram
from pydantic import BaseModel, Field, create_model
from llama_index.llms.openai import OpenAI
import re


# class StructuredQueryEngineBuilder:

mock_data = [
{
"attribute_id": "bg6t6xhlfxbw",
"name": "Weight",
"description": "The weight of the item's frame",
"unit": "pounds",
"format": "number",
"options": [],
"attribute_group_id": "askfdjs;lj",
"attribute_group_name": "Model Information"
},
{
"attribute_id": "bnccqpc9nwtd",
"name": "Has safety clutch",
"description": "Does the item have a safety clutch, true or false",
"unit": "",
"format": "boolean",
"options": [],
"attribute_group_id": "askfdjsalj",
"attribute_group_name": "Model Information"
}
]

## Helper function to create model friendly names
## This might need to return a tuple or hashmap so we can map the model names back to their original names later on, but I'm not going to overcomplicate it for now
def clean_attr_name(name) -> str:
    ## Convert to lowercase
    name = name.lower()
    ## Replace spaces and parentheses with underscores
    name = re.sub(r'[\s\(\)]', '_', name)
    ## Remove any non-alphanumeric characters (except underscores)
    name = re.sub(r'[^\w]', '', name)
    ## Remove trailing underscores
    name = name.strip('_')
    return name

def create_dynamic_attr_model(attributes: list[dict], model_name: str = "1324D_Standard_Trencher") -> type[BaseModel]:
    ## We need to pass the llm an empty model with fields to populate for:
    ## each attribute name  with the provided description, or at least a pretty good default string to help the llm
    ## We will need to clean up the names so they can be used as model parameters. No spaces, parens, etc
    ## Unit & Format can be bundled into the description if they exist
    ## Options is a weird one - it's hard to nail down what this means for the llm call exactly. Need clarification from Garrett.

    ## The llm must also know the make & model of the item we are interested in, but that should be in the prompt from the query engine rather than the pydantic model

    ## We care about attribute_id because we need to keep those things linked.

    attr_fields = {
        clean_attr_name(attr['name']): (
            str,
            make_description(
                name=attr['name'],
                description=attr.get('description'),
                unit=attr.get('unit'),
                format=attr.get('format')
            )
        )
        for attr in attributes
    }

    dynamic_model = create_model(model_name, **attr_fields)
    return dynamic_model

## Helper function to create descriptions for the model fields to assist the LLM call
def make_description(name: str, description: str = None, unit: str = None, format: str = None) -> Field:
    ## Start with base description.
    ## If a description is provided, default to that. Else create a description string using the name of the attribute
    if description:
        base_desc = description
    else:
        base_desc = f"The {name.lower()} of the item, if present"

    ## If we have format and/or unit info, we'll create an additional_info string.
    ## (We're not doing anything with options yet - this might live here as well, or in the llm prompt with the make and model info)
    additional_info = []
    if format:
        additional_info.append(f"a {format}")
    if unit:
        additional_info.append(f"in {unit}")

    ## Combine description with additional info if any exists
    if additional_info:
        final_desc = f"{base_desc}. The value should be {' '.join(additional_info)}."
    else:
        final_desc = f"{base_desc}."

    ## Pydantic models need a default for a basic instantiation or else they will throw
    return Field(description=final_desc, default=None)

def query(index: VectorStoreIndex):

    structured_attr_model = create_dynamic_attr_model(mock_data)
    print("DYNAMIC MODEL CREATED")
    print(structured_attr_model)

    llm = OpenAI(model="gpt-4o-mini")
    query_engine = index.as_query_engine(output_cls=structured_attr_model, ressponse_mode="compact", llm=llm)
    print("QUERY ENGINE CREATED")
    response = query_engine.query("Add attributes to model if found, or None for the specific model 1324D Standard Trencher")

    print("RESPONSE")
    print(response)

