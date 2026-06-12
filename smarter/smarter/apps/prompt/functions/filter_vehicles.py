# This module was developed with the assistance of Claude (claude.ai), Anthropic's AI assistant.
"""
This module provides a vehicle inventory filtering function for use with
Smarter's extension of the OpenAI API function calling feature.

Following the same protocol as ``protocol.py`` (the weather tool template),
this module implements the ``filter_vehicles`` tool — Tool 1 of 4 in the
agentic car-buying workflow. It serves as a concrete example of applying
the Smarter tool protocol to a static-data use case.

Best practices followed from the protocol template include:

- Robust input validation using Pydantic models, with clear error messages for
  invalid inputs.
- Fully annotated function signatures with type hints for all variables,
  parameters, and return types.
- Comprehensive error handling with specific exceptions for different failure
  modes, and logging of error details.
- Use of a WaffleSwitchedLoggerWrapper to enable or disable logging based on a
  lambda function, allowing for dynamic control of logging without code changes.
- Detailed Sphinx-compatible docstrings for all functions and classes,
  following the NumPy style guide, including sections for Parameters, Returns,
  Raises, Examples, and See Also.
- Use of Django signals to emit events at key points in the function execution,
  allowing for extensibility and integration with other parts of the application.

See Also
--------
https://developers.openai.com/api/docs/guides/function-calling

Protocol
--------
Two functions are required for implementing this protocol:

1. A factory function that returns a JSON-compatible dictionary defining the
   tool for OpenAI LLM function calling.
2. A function that implements the desired behaviour (in this case, filtering
   a vehicle inventory by body type and budget).

The factory function (``filter_vehicles_tool_factory``) is consumed by
``OpenAISmarterClient.handle_function_provided()`` during the tool
registration process, which adds the tool definition to the list of available
tools for the LLM to call.

The implementation function (``filter_vehicles``) is executed by
``OpenAISmarterClient.process_tool_call()`` when the LLM requests this tool
following iteration #1 of the conversation.

Overview
--------
Filters a static vehicle inventory dataset and returns all vehicles that match
a specified body type and fall within the user's budget. Optionally constrains
results to a minimum model year. Results are sorted by price ascending.

Dependencies
------------
* pydantic
* smarter (platform)

Signals
-------
* llm_tool_presented
* llm_tool_requested
* llm_tool_responded
"""

# standard Python library imports
import logging
from datetime import datetime
from typing import Any, Optional

# OpenAI tool call type
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

# Smarter platform imports
from smarter.apps.prompt.signals import (
    llm_tool_presented,
    llm_tool_requested,
    llm_tool_responded,
)
from smarter.common.helpers.console_helpers import formatted_text
from smarter.lib import json
from smarter.lib.logging import WaffleSwitchedLoggerWrapper

# local imports
# from .models import FilterVehiclesRequestModel
from .utils import should_log

# ---------------------------------------------------------------------------
# Module-level logger setup
# ---------------------------------------------------------------------------
base_logger = logging.getLogger(__name__)
logger = WaffleSwitchedLoggerWrapper(base_logger, should_log)
logger_prefix = formatted_text(__name__ + ".filter_vehicles()")

# ---------------------------------------------------------------------------
# Static vehicle inventory dataset
# (stand-in for a real database or REST API in this prototype)
# ---------------------------------------------------------------------------
VEHICLE_INVENTORY: list[dict[str, Any]] = [
    {"make": "Toyota",    "model": "RAV4",       "year": 2025, "price": 34500, "body_type": "SUV"},
    {"make": "Honda",     "model": "CR-V",        "year": 2025, "price": 37900, "body_type": "SUV"},
    {"make": "Kia",       "model": "Telluride",   "year": 2025, "price": 44200, "body_type": "SUV"},
    {"make": "Ford",      "model": "Escape",      "year": 2025, "price": 31000, "body_type": "SUV"},
    {"make": "Subaru",    "model": "Forester",    "year": 2025, "price": 33500, "body_type": "SUV"},
    {"make": "Mazda",     "model": "CX-5",        "year": 2025, "price": 36000, "body_type": "SUV"},
    {"make": "Chevrolet", "model": "Equinox",     "year": 2025, "price": 32000, "body_type": "SUV"},
    {"make": "Nissan",    "model": "Rogue",        "year": 2025, "price": 35000, "body_type": "SUV"},
    {"make": "BMW",       "model": "X5",           "year": 2025, "price": 68000, "body_type": "SUV"},
    {"make": "Ford",      "model": "Explorer",     "year": 2024, "price": 46000, "body_type": "SUV"},
    {"make": "Toyota",    "model": "Camry",        "year": 2025, "price": 28000, "body_type": "Sedan"},
    {"make": "Honda",     "model": "Accord",       "year": 2024, "price": 27000, "body_type": "Sedan"},
]


# ---------------------------------------------------------------------------
# Pydantic request model  (mirrors WeatherRequestModel in the template)
# ---------------------------------------------------------------------------
from pydantic import BaseModel, Field, field_validator  # noqa: E402


class FilterVehiclesRequestModel(BaseModel):
    """
    Pydantic model for validating ``filter_vehicles`` input arguments.

    Attributes
    ----------
    budget : int
        Maximum purchase price in USD. Must be a positive integer.
    body_type : str
        Vehicle body style to filter on (e.g. ``"SUV"``). Case-insensitive.
    year_min : int, optional
        Earliest model year to include. Defaults to ``current_year - 1``.
    """

    budget: int = Field(..., gt=0, description="Maximum price in USD, e.g. 45000")
    body_type: str = Field(..., min_length=1, description="Vehicle body style, e.g. 'SUV'")
    year_min: Optional[int] = Field(
        default=None,
        ge=1900,
        le=datetime.now().year + 1,
        description="Earliest model year (inclusive)",
    )

    @field_validator("body_type")
    @classmethod
    def strip_body_type(cls, v: str) -> str:
        """Strip whitespace and normalise to title-case."""
        return v.strip().title()


# ---------------------------------------------------------------------------
# 1. Factory function
# ---------------------------------------------------------------------------

def filter_vehicles_tool_factory() -> dict[str, Any]:
    """
    Constructs and returns a JSON-compatible dictionary defining the
    ``filter_vehicles`` tool for OpenAI LLM function calling.

    See Also
    --------
    https://developers.openai.com/api/docs/guides/function-calling

    Django Signals
    --------------
    - llm_tool_presented : Sent when this function is called.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the tool definition for ``filter_vehicles``,
        formatted for OpenAI LLM function calling.

    Examples
    --------
    >>> tool_def = filter_vehicles_tool_factory()
    >>> tool_def["type"]
    'function'
    >>> tool_def["function"]["name"]
    'filter_vehicles'
    """
    llm_tool_presented.send(
        sender=filter_vehicles_tool_factory,
        tool={
            "name": filter_vehicles.__name__,
            "description": (
                "Search a vehicle inventory for cars matching a given body "
                "type and budget. Returns a list of matching vehicles."
            ),
        },
    )

    tool: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": filter_vehicles.__name__,
            "description": (
                "Search a vehicle inventory for cars matching a given body "
                "type and budget. Returns a list of matching make/model/year/"
                "price objects sorted by price ascending."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "budget": {
                        "type": "integer",
                        "description": "Maximum purchase price in USD (e.g. 45000).",
                    },
                    "body_type": {
                        "type": "string",
                        "description": "Vehicle body style to filter on, e.g. 'SUV'.",
                    },
                    "year_min": {
                        "type": "integer",
                        "description": (
                            "Earliest model year to include. "
                            "Defaults to the current year minus 1 if not provided."
                        ),
                    },
                },
                "required": ["budget", "body_type"],
                "additionalProperties": False,
            },
        },
    }
    return tool


# ---------------------------------------------------------------------------
# 2. Implementation function
# ---------------------------------------------------------------------------

def filter_vehicles(tool_call: ChatCompletionMessageToolCall) -> list[dict[str, Any]]:
    """
    Filters the static vehicle inventory by body type and budget, returning
    all matching vehicles sorted by price ascending. The basic flow is:

    1. Define and initialise variables to be used in the function.
    2. Parse and validate the input arguments from the tool call using
       the ``FilterVehiclesRequestModel`` Pydantic model.
    3. Apply body-type and budget filters to ``VEHICLE_INVENTORY``.
    4. Apply the optional ``year_min`` constraint.
    5. Handle the no-results edge case with a helpful suggestion.
    6. Sort results by price and package them for return.
    7. Return the result as a JSON list (to be compatible with OpenAI
       function calling response format).

    Parameters
    ----------
    tool_call : ChatCompletionMessageToolCall
        The OpenAI tool call object containing the function name and
        JSON-encoded arguments supplied by the LLM.

    Django Signals
    --------------
    - llm_tool_requested : Sent when the tool is called, with the tool call
      data, budget, and body_type.
    - llm_tool_responded : Sent after the tool has generated a response, with
      the tool call data and the response payload.

    Returns
    -------
    list[dict[str, Any]]
        A single-element JSON list containing either:

        - A success payload with keys ``vehicles`` (list), ``count`` (int).
        - A no-results payload with keys ``vehicles`` (empty list),
          ``count`` (0), and ``message`` (str) suggesting a budget increase.
        - An error payload with key ``error`` (str) describing the failure.

    Raises
    ------
    Does not raise. All exceptions are caught, logged, and returned as
    structured error dictionaries.

    Examples
    --------
    Typical usage (simulated tool call):

    >>> import json as _json
    >>> from types import SimpleNamespace
    >>> fake_call = SimpleNamespace(
    ...     function=SimpleNamespace(
    ...         arguments=_json.dumps({"budget": 45000, "body_type": "SUV"})
    ...     )
    ... )
    >>> result = filter_vehicles(fake_call)
    >>> result[0]["count"]
    8

    See Also
    --------
    filter_vehicles_tool_factory : Registers this function with the LLM.
    """

    # -------------------------------------------------------------------------
    # 1.) Define and annotate variables used in this function.
    # -------------------------------------------------------------------------

    # parsed and validated request fields
    budget: int = 0
    body_type: str = ""
    year_min: int = datetime.now().year - 1

    # filtered vehicle list and final result
    filtered: list[dict[str, Any]] = []
    result: dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # 2.) Parse and validate input arguments from the tool call.
    # -------------------------------------------------------------------------
    if not tool_call or not tool_call.function or not tool_call.function.arguments:
        return [{"error": "No arguments provided. Please supply a budget and body_type."}]

    try:
        raw_args = tool_call.function.arguments
        if isinstance(raw_args, str):
            raw_args = json.loads(raw_args)

        # Validate with the Pydantic model — raises ValidationError on bad input.
        request = FilterVehiclesRequestModel(**raw_args)

        budget    = request.budget
        body_type = request.body_type          # already title-cased by validator
        year_min  = request.year_min if request.year_min is not None else year_min

        logger.debug(f"{logger_prefix} Parsed budget={budget}, body_type={body_type!r}, year_min={year_min}")

    except Exception as e:
        return [{"error": f"Invalid arguments: {e}"}]

    # Send a Django signal that the tool was requested.
    llm_tool_requested.send(
        sender=filter_vehicles,
        tool_call=tool_call.model_dump(),
        budget=budget,
        body_type=body_type,
        year_min=year_min,
    )

    # -------------------------------------------------------------------------
    # 3.) & 4.) Apply body-type, budget, and year filters.
    # -------------------------------------------------------------------------
    try:
        filtered = [
            vehicle for vehicle in VEHICLE_INVENTORY
            if vehicle["body_type"].title() == body_type
            and vehicle["price"] <= budget
            and vehicle["year"] >= year_min
        ]
        logger.debug(f"{logger_prefix} {len(filtered)} vehicle(s) matched filters.")

    except (KeyError, TypeError, ValueError) as e:
        msg = f"Error filtering vehicle inventory: {e}"
        logger.error(f"{logger_prefix} {msg}")
        return [{"error": msg}]
    except Exception as e:
        msg = f"Unexpected error filtering vehicle inventory: {e}"
        logger.error(f"{logger_prefix} {msg}")
        return [{"error": msg}]

    # -------------------------------------------------------------------------
    # 5.) Handle no-results edge case.
    # -------------------------------------------------------------------------
    if not filtered:
        result = {
            "vehicles": [],
            "count": 0,
            "message": (
                f"No {body_type}s found under ${budget:,}. "
                f"Consider raising your budget by $5,000 or broadening the search."
            ),
        }
        logger.debug(f"{logger_prefix} No results — returning helpful message.")
        return [result]

    # -------------------------------------------------------------------------
    # 6.) Sort by price ascending and build the result payload.
    # -------------------------------------------------------------------------
    try:
        filtered.sort(key=lambda v: v["price"])
        result = {
            "vehicles": filtered,
            "count": len(filtered),
        }
    except Exception as e:
        msg = f"Error sorting vehicle results: {e}"
        logger.error(f"{logger_prefix} {msg}")
        return [{"error": msg}]

    # Send a Django signal that the tool has generated a response.
    llm_tool_responded.send(
        sender=filter_vehicles,
        tool_call=tool_call.model_dump(),
        tool_response=result,
    )

    # -------------------------------------------------------------------------
    # 7.) Return the result as a JSON list (OpenAI function calling format).
    # -------------------------------------------------------------------------
    return [result]
