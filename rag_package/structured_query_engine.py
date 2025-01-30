from llama_index.core import VectorStoreIndex
from pydantic import BaseModel, Field, create_model
from llama_index.llms.openai import OpenAI
from typing import Optional
import re

class StructuredQueryEngineBuilder:
    def __init__(self, make: str, model: str, attributes: list[dict],
                 prompt: str, llm_model: str):
        self.make = make
        self.model = model
        self.attributes = attributes
        self.prompt = prompt
        self.llm_model = llm_model

    def _clean_attr_name(self, name: str) -> str:
        ## Remove special characters from pydantic model field names
        name = name.lower()
        name = re.sub(r'[\s\(\)]', '_', name)
        name = re.sub(r'[^\w]', '', name)
        name = name.strip('_')
        return name

    def _make_description(self, name: str, description: str = None,
                          unit: str = None, format: str = None,
                          vocabulary_options: list = None) -> Field:
        ## This is the description the LLM will use to find the appropriate attribute
        ## If descriptions are provided use those
        if description:
            base_desc = description
        ## Otherwise we'll use the following default prompt
        else:
            base_desc = f"The {name.lower()} of the item"

        additional_info = []

        ## Add format information if provided
        if format:
            additional_info.append(f"Should be a {format}")

        ## Add unit information if provided also
        if unit:
            additional_info.append(f"in {unit}")

        ## Add vocabulary options if provided. These are a multichoice selection for the LLM
        ## These are essentially saying "respond only with one of the following answers"
        ## "NO VALUE FOUND" is due to the LLM often defaulting to "false" if it can't find the information, which might not be accurate.
        if vocabulary_options:
            options_str = ", ".join(vocabulary_options)
            additional_info.append(f"Must be one of: [{options_str}]")
            additional_info.append("Use 'NO VALUE FOUND' if cannot be determined from the available data")

        if additional_info:
            final_desc = f"{base_desc}. {' '.join(additional_info)}."
        else:
            final_desc = f"{base_desc}."

        return Field(description=final_desc, default=None)

    def create_dynamic_attr_model(self, model_name: str = "DynamicModel") -> type[BaseModel]:
        attr_fields = {}

        for attr in self.attributes:
            clean_name = self._clean_attr_name(attr['name'])

            # All fields are strings to handle various types uniformly
            attr_fields[clean_name] = (
                Optional[str],
                self._make_description(
                    name=attr['name'],
                    description=attr.get('description'),
                    unit=attr.get('unit'),
                    format=attr.get('format'),
                    vocabulary_options=attr.get('vocabulary_options')
                )
            )

        return create_model(model_name, **attr_fields)

    def query_one(self, index: VectorStoreIndex) -> dict[str, list[str]]:
        ## Although this is for a single equipment item query, we still return the attribute values in a list (like we do for query_all_list)
        ## This is to keep the pipeline consistent.
        ## Regardless of the type of query, the resulting dictionary can be fed directly into the CSVCreator's export_dict_to_csv method
        structured_attr_model = self.create_dynamic_attr_model()

        llm = OpenAI(model=self.llm_model)
        query_engine = index.as_query_engine(
            output_cls=structured_attr_model,
            response_mode="compact",
            llm=llm
        )

        response = query_engine.query(self.prompt)

        ## Added a "retry once" policy if the LLM gives us back a string instead of a pydantic model
        if isinstance(response, str):
            retry_query = f"{self.prompt} Do not return a string response - only return the requested attribute values."
            response = query_engine.query(retry_query)
            if isinstance(response, str):
                raise ValueError("LLM repeatedly returned string response instead of structured data")

        ## Convert returning Pydantic model back to a dictionary
        model_response = response.response
        result_dict: dict[str, list[str]] = {
            'make': [self.make],
            'model': [self.model]
        }

        ## We will return a dictionary instead of a model, because the model attribute names would be in snake case
        for field in model_response.model_fields:
            value = getattr(model_response, field)

            # Find the original attribute name and info
            attr_info = next((attr for attr in self.attributes
                              if self._clean_attr_name(attr['name']) == field), None)
            original_name = attr_info['name'] if attr_info else field

            ## we want to concatenate the value + unit when it makes sense
            ## For instance "900 pounds" makes sense
            ## but "true boolean", or "NO VALUE FOUND string" don't
            is_boolean = (
                    value in ('true', 'false', 'yes', 'no') or
                    (attr_info and attr_info.get('vocabulary_options') == ['yes', 'no'])
            )

            if value is None:
                formatted_value = "NO VALUE FOUND"
            elif is_boolean:
                formatted_value = str(value).lower()
            elif attr_info and attr_info.get('unit') and value != "NO VALUE FOUND":
                formatted_value = f"{value} {attr_info['unit']}"
            else:
                formatted_value = str(value)

            result_dict[original_name.lower()] = [formatted_value]

        return result_dict

    def query_many(self, index: VectorStoreIndex, make_model_pairs: list[tuple[str, str]]):
        ## Here we query the attributes for multiple equipment items (like for a cat class)
        ## We take the make model pairs as tuples instead of a dict, because in a dict because equipment items of the same make overwrite the others
        ## This method can probably be optimized by making it async and having the calls run in parallel - but one step at a time

        ## First we need to set up the dictionary that will hold the outputs
        result_dict = {
            'make': [],
            'model': [],
            **{attr['name'].lower(): [] for attr in self.attributes}
        }

        ## Then here's the simple query method which makes one call at a time (for now)
        for make, model in make_model_pairs:
            try:
                self.make = make
                self.model = model
                single_result = self.query_one(index)

                ## Add the result of each call to our dictionary
                for key in result_dict:
                    value = single_result.get(key, ["NO VALUE FOUND"])[0]
                    result_dict[key].append(value)

            except Exception as e:
                print(f"Error querying {make} {model}: {str(e)}")
                ## Add error values to make debugging easier
                for key in result_dict:
                    if key == 'make':
                        result_dict[key].append(make)
                    elif key == 'model':
                        result_dict[key].append(model)
                    else:
                        result_dict[key].append(f"ERROR: {str(e)}")

        return result_dict