from llama_index.core import VectorStoreIndex
from pydantic import BaseModel, Field, create_model
from llama_index.llms.openai import OpenAI
from typing import Optional
import re
import asyncio

class StructuredQueryEngineBuilder:
    def __init__(self, index: VectorStoreIndex, llm_model: str):
        ## Generic initialization with the vector store and an LLM
        ## The LLM dependency could be moved to the query argument list, but I think it's reasonable to keep it here until that becomes a necessity.
        self.index = index
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
        ## Since these fields are optional, we need to compose the description differently depending on what is passed in

        ## If descriptions are provided, just use those
        if description:
            base_desc = description
        else:
            ## Otherwise we'll use the following default prompt
            base_desc = f"The {name.lower()} of the item"

        additional_info = []
        if format:
        ## Add format information if provided
            additional_info.append(f"Should be a {format}")
        if unit:
        ## Add unit information if provided also
            additional_info.append(f"in {unit}")
        if vocabulary_options:
        ## Add vocabulary options if provided. These are a multichoice selection for the LLM
        ## These are essentially saying "respond only with one of the following answers"
        ## Due to the LLM often defaulting to "false" if it can't find the information (which might not be accurate) we ask it to instead respond with: "NO VALUE FOUND"
            options_str = ", ".join(vocabulary_options)
            additional_info.append(f"Must be one of: [{options_str}]")
            additional_info.append("Use 'NO VALUE FOUND' if cannot be determined from the available data")
        ## And finally return the most detailed version of the description we can
        if additional_info:
            final_desc = f"{base_desc}. {' '.join(additional_info)}."
        else:
            final_desc = f"{base_desc}."

        return Field(description=final_desc, default=None)

    def create_dynamic_attr_model(self, attributes: list[dict], model_name: str = "DynamicModel") -> type[BaseModel]:
        ## Creates a pydantic model based on the attributes for each make and model, with descriptions for each field
        ## Multiple queries can use the same BaseModel, so long as each receives make / model information in the prompt (see line 108)
        attr_fields = {}

        for attr in attributes:
            clean_name = self._clean_attr_name(attr['name'])
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

    def query(self,
              make_model_pairs: list[tuple[str, str]],
              attributes: list[dict],
              prompt_template: str) -> dict[str, list[str]]:

        ## This query method is more generic and flexible, and can now be used for CatClasses, so long as each member of the cat class is represented as an Equipment Item
        ## Note that we take the make model pairs as tuples instead of a dict, because in a dict because equipment items of the same make overwrite the others
        ## This method can be optimized by making it async and having the calls run in parallel - that's the next step

        ## First we need to set up the dictionary that will hold the outputs
        result_dict = {
            'make': [],
            'model': [],
            **{self._clean_attr_name(attr['name']): [] for attr in attributes}
        }

        ## Then we build our query engine
        ## Because we're doing a structured LLM call, the query engine needs to be initialized with a single Pydantic model
        ## Even though each query is for a different object (make and model now - but CAT class in the future), we use the prompt to pass this information into the LLM call
        structured_attr_model = self.create_dynamic_attr_model(attributes)
        llm = OpenAI(model=self.llm_model)
        query_engine = self.index.as_query_engine(
            output_cls=structured_attr_model,
            response_mode="compact",
            llm=llm
        )

        ## Then here's the simple query method which makes one call at a time (for now)
        for make, model in make_model_pairs:
            try:
                ## Format prompt for this specific make/model
                prompt = prompt_template.format(make=make, model=model)

                ## Query the index
                response = query_engine.query(prompt)

                ## Handle unwanted string responses with a one time retry (easier than trying to parse out what we need)
                if isinstance(response, str):
                    retry_prompt = f"{prompt} Do not return a string response - only return the requested attribute values."
                    response = query_engine.query(retry_prompt)
                    if isinstance(response, str):
                        raise ValueError("LLM Error: LLM repeatedly returned string response instead of structured data expected from structured LLM call")

                ## Extract the response
                model_response = response.response

                # Add make and model
                result_dict['make'].append(make)
                result_dict['model'].append(model)

                ## Clean up and format the returned values
                ## This stops us from exporting weird cells in our CSV like "true boolean" or "NO VALUE FOUND string"
                for field_name in result_dict:
                    if field_name not in ['make', 'model']:
                        value = getattr(model_response, field_name, None)

                        # Find the original attribute info
                        attr_info = next((attr for attr in attributes
                                          if self._clean_attr_name(attr['name']) == field_name), None)

                        # Format the value
                        if value is None:
                            formatted_value = "NO VALUE FOUND"
                        elif isinstance(value, bool) or (
                                attr_info and attr_info.get('vocabulary_options') == ['yes', 'no']):
                            formatted_value = str(value).lower()
                        elif attr_info and attr_info.get('unit') and value != "NO VALUE FOUND":
                            if attr_info['unit'] != ("boolean" or "bool" or "string"):
                                formatted_value = f"{value} {attr_info['unit']}"
                            else:
                                formatted_value = f"{value}"
                        else:
                            formatted_value = str(value)

                        result_dict[field_name].append(formatted_value)

            except Exception as e:
                print(f"Error querying {make} {model}: {str(e)}")
                ## Add error values for ALL fields to maintain list lengths - otherwise both this method and the downstream CSVCreator step will fail
                result_dict['make'].append(make)
                result_dict['model'].append(model)
                for field_name in result_dict:
                    if field_name not in ['make', 'model']:
                        result_dict[field_name].append(f"ERROR: {str(e)}")

        list_lengths = {len(v) for v in result_dict.values()}
        if len(list_lengths) != 1:
            raise ValueError(
                f"Inconsistent list lengths in result: {dict((k, len(v)) for k, v in result_dict.items())}")

        return result_dict

    async def aquery(self,
                    make_model_pairs: list[tuple[str, str]],
                    attributes: list[dict],
                    prompt_template: str) -> dict[str, list[str]]:

        ## First we need to set up the dictionary that will hold the outputs
        result_dict = {
            'make': [],
            'model': [],
            **{self._clean_attr_name(attr['name']): [] for attr in attributes}
        }

        ## Then we build our query engine
        ## Because we're doing a structured LLM call, the query engine needs to be initialized with a single Pydantic model
        ## Even though each query is for a different object (make and model now - but CAT class in the future), we use the prompt to pass this information into the LLM call
        structured_attr_model = self.create_dynamic_attr_model(attributes)
        llm = OpenAI(model=self.llm_model)
        query_engine = self.index.as_query_engine(
            output_cls=structured_attr_model,
            response_mode="compact",
            llm=llm
        )

        async def process_single_query(make: str, model: str) -> dict:
            try:
                ## Format prompt for this specific make/model
                prompt = prompt_template.format(make=make, model=model)

                ## Query the index - Note: using await since we're in an async context
                response = await query_engine.aquery(prompt)

                ## Handle unwanted string responses with a one time retry
                if isinstance(response, str):
                    retry_prompt = f"{prompt} Do not return a string response - only return the requested attribute values."
                    response = await query_engine.aquery(retry_prompt)
                    if isinstance(response, str):
                        raise ValueError(
                            "LLM Error: LLM repeatedly returned string response instead of structured data expected from structured LLM call")

                ## Extract the response
                model_response = response.response

                # Create a result dictionary for this specific query
                single_result = {
                    'make': make,
                    'model': model
                }

                ## Clean up and format the returned values
                for field_name in result_dict:
                    if field_name not in ['make', 'model']:
                        value = getattr(model_response, field_name, None)

                        # Find the original attribute info
                        attr_info = next((attr for attr in attributes
                                          if self._clean_attr_name(attr['name']) == field_name), None)

                        # Format the value (same formatting logic as before)
                        if value is None:
                            formatted_value = "NO VALUE FOUND"
                        elif isinstance(value, bool) or (
                                attr_info and attr_info.get('vocabulary_options') == ['yes', 'no']):
                            formatted_value = str(value).lower()
                        elif attr_info and attr_info.get('unit') and value != "NO VALUE FOUND":
                            if attr_info['unit'] != ("boolean" or "bool" or "string"):
                                formatted_value = f"{value} {attr_info['unit']}"
                            else:
                                formatted_value = f"{value}"
                        else:
                            formatted_value = str(value)

                        single_result[field_name] = formatted_value

                return single_result

            except Exception as e:
                ## Add error values for ALL fields to maintain list lengths - otherwise both this method and the downstream CSVCreator step will fail
                return {
                    'make': make,
                    'model': model,
                    **{field_name: f"ERROR: {str(e)}"
                       for field_name in result_dict if field_name not in ['make', 'model']}
                }

        ## Create tasks for all make/model pairs
        tasks = [process_single_query(make, model) for make, model in make_model_pairs]

        ## Execute and gather all queries concurrently
        results = await asyncio.gather(*tasks)

        ## Combine all results into the final dictionary
        for result in results:
            for field_name in result_dict:
                result_dict[field_name].append(result[field_name])

        ## Double check we nothing went wrong with our dictionary
        list_lengths = {len(v) for v in result_dict.values()}
        if len(list_lengths) != 1:
            raise ValueError(
                f"Inconsistent list lengths in result: {dict((k, len(v)) for k, v in result_dict.items())}")

        return result_dict