from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import (
    PromptBase,
    UpdatePromptRequest,
    PromptResponse,
    PromptListResponse
)
from app.services.prompts import (
    list_prompts,
    create_prompt,
    update_prompt,
    delete_prompt,
)

router = APIRouter(prefix="/prompts", tags=["prompts"])


# List all prompts with filters
@router.get("/list_available_prompts", response_model=PromptListResponse)
def list_prompts_api(
    type: str | None = Query(None, description="Filter by type: generator or validator"),
    user_id: str | None = Query(None, description="Filter by user_id"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """List all prompts with optional filters."""
    prompts, total = list_prompts(
        prompt_type=type,
        user_id=user_id,
        limit=limit,
        offset=offset
    )
    return PromptListResponse(prompts=prompts, total=total)


# Create a new prompt
@router.post("/append_new_prompt", response_model=PromptResponse, status_code=201)
def create_prompt_api(req: PromptBase):
    """Create and add a new prompt."""
    try:
        prompt = create_prompt(
            prompt_type=req.type,
            template=req.template,
            user_id=req.user_id
        )
        return prompt
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Update an existing prompt
@router.patch("/update_existing_prompt", response_model=PromptResponse)
def update_prompt_api(
    prompt_id: int = Query(..., description="ID of the prompt to update"),
    req: UpdatePromptRequest = None
):
    """Update an existing prompt using its ID."""
    try:
        prompt = update_prompt(
            prompt_id=prompt_id,
            template=req.template if req else None
        )
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        return prompt
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Delete a prompt
@router.delete("/delete_prompt")
def delete_prompt_api(
    prompt_id: int = Query(..., description="ID of the prompt to delete")
):
    """Delete a prompt using its ID."""
    deleted = delete_prompt(prompt_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"status": "deleted", "id": prompt_id}
