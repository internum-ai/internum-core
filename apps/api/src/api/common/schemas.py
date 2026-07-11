from typing import Any

from pydantic import BaseModel, Field


class ErrorDetailSchema(BaseModel):
    code: str = Field(description="Machine-readable error code, e.g. 'invalid_api_key'.")
    message: str = Field(description="Human-readable description of the error.")
    details: dict[str, Any] | list[Any] | None = Field(
        default=None, description="Additional structured context about the error, if any."
    )
    request_id: str | None = Field(
        default=None,
        alias="requestId",
        description="Identifier of the request that produced this error, for log correlation.",
    )


class ErrorResponseSchema(BaseModel):
    error: ErrorDetailSchema = Field(description="The error envelope returned for failed requests.")
