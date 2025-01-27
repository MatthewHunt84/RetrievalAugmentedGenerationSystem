from llama_index.core import VectorStoreIndex
from pydantic import BaseModel, Field, create_model
from llama_index.llms.openai import OpenAI
import re
import json


class AttributeMapper:
    def __init__(self, attributes: list[dict]):
        ## Store original attributes for later reference
        self.original_attributes = attributes
        self.name_to_id = {}
        self.name_to_group = {}
        self.name_to_original = {}

        for attr in attributes:
            cleaned_name = self._clean_attr_name(attr['name'])
            self.name_to_id[cleaned_name] = attr['attribute_id']
            self.name_to_group[cleaned_name] = attr['attribute_group_id']
            self.name_to_original[cleaned_name] = attr['name']

    def _clean_attr_name(self, name: str) -> str:
        name = name.lower()
        name = re.sub(r'[\s\(\)]', '_', name)
        name = re.sub(r'[^\w]', '', name)
        name = name.strip('_')
        return name

    def enrich_response(self, llm_response: dict) -> list[dict]:
        ## We don't want to bloat the LLM call with things like attribute_ids which use up tokens but aren't useful.
        ## So we enrich the attribute data with the llm responses here after the call.
        enriched_response = []

        for clean_name, value in llm_response.items():
            enriched_response.append({
                'attribute_id': self.name_to_id.get(clean_name),
                'attribute_group_id': self.name_to_group.get(clean_name),
                'attribute': self.name_to_original.get(clean_name),
                'value': value if value is not None else "NO VALUE FOUND"
            })

        return enriched_response


class StructuredQueryEngineBuilder:
    def __init__(self, make: str, model: str, attributes: list[dict],
                 prompt: str, llm_model: str):
        self.make = make
        self.model = model
        self.attributes = attributes
        self.attribute_mapper = AttributeMapper(attributes)
        self.prompt = prompt
        self.llm_model = llm_model

    def _clean_attr_name(self, name: str) -> str:
        return self.attribute_mapper._clean_attr_name(name)

    def _make_description(self, name: str, description: str = None,
                          unit: str = None, format: str = None) -> Field:
        if description:
            base_desc = description
        else:
            base_desc = f"The {name.lower()} of the item, if present"

        additional_info = []
        if format:
            additional_info.append(f"a {format}")
        if unit:
            additional_info.append(f"in {unit}")

        if additional_info:
            final_desc = f"{base_desc}. The value should be {' '.join(additional_info)}."
        else:
            final_desc = f"{base_desc}."

        return Field(description=final_desc, default=None)

    def create_dynamic_attr_model(self, model_name: str = "DynamicModel") -> type[BaseModel]:
        ## Model name is meaningless, but we must have something for the pydantic create_model method
        attr_fields = {
            self._clean_attr_name(attr['name']): (
                str,  # Use str for all fields to handle various types uniformly
                self._make_description(
                    name=attr['name'],
                    description=attr.get('description'),
                    unit=attr.get('unit'),
                    format=attr.get('format')
                )
            )
            for attr in self.attributes
        }

        return create_model(model_name, **attr_fields)

    def query(self, index: VectorStoreIndex) -> list[dict]:
        structured_attr_model = self.create_dynamic_attr_model()

        llm = OpenAI(model=self.llm_model)
        query_engine = index.as_query_engine(
            output_cls=structured_attr_model,
            response_mode="compact",
            llm=llm
        )

        ## Sometimes the LLM returns a string instead of a model. We are building in a single retry before raising an error
        response = query_engine.query(self.prompt)

        if isinstance(response, str):
            ## Retry with prompt being even more explicit about not returning a string.
            ## We could put this in all caps maybe - but that feels like yelling at a robot
            retry_prompt = f"{self.prompt} Do not return a string response - only return the requested attribute values."
            response = query_engine.query(retry_prompt)

            ## If still getting a string, it means the LLM sent back strings twice in a row. We'll raise an error here
            if isinstance(response, str):
                raise ValueError("LLM repeatedly returned string response instead of structured data")

        ## Next we're going to map the LLM responses back to their attribute IDs, before returning that object
        response_dict = (
            response.dict() if hasattr(response, 'dict')
            else response if isinstance(response, dict)
            else json.loads(str(response))
        )

        enriched_response = self.attribute_mapper.enrich_response(response_dict)

        return enriched_response

