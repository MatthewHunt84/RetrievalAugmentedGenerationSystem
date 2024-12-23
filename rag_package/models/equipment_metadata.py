"""
Domain models for equipment metadata in the RAG pipeline.

This module defines the core data structures for representing equipment metadata.
These models are used throughout the pipeline, particularly in conjunction with
the metadata extraction process.
"""
from typing import Optional
from pydantic import BaseModel, Field


class EquipmentMetadata(BaseModel):
    """
    Metadata schema for equipment product information.

    This schema defines the structure of metadata extracted from technical documentation.
    It is designed to work in conjunction with the metadata extraction prompts
    defined in MetadataExtractionConfig.
    """
    product_name: str = Field(
        ...,
        description="The full name of the product"
    )
    model_number: str = Field(
        ...,
        description="The specific model number/identifier of the product"
    )
    manufacturer: str = Field(
        ...,
        description="The manufacturer of the equipment"
    )
    category: str = Field(
        ...,
        description="Main equipment category"
    )
    subcategory: Optional[str] = Field(
        None,
        description="More specific classification within the main category"
    )
    year: Optional[str] = Field(
        None,
        description="Manufacturing or model year if available"
    )
    specifications: list[str] = Field(
        description="Specifications for this model such as size and weight"
    )
    capabilities: list[str] = Field(
        description="Any notable capabilities or features which would differentiate this model from similar models"
    )
    content_types: list[str] = Field(
        description="Types of content present for this model"
    )