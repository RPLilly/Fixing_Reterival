from __future__ import annotations

from fastapi import APIRouter

from app.services.qa_langgraph import answer_langgraph_hybrid
from app.models.schemas import AnswerLanggraphRequest

router = APIRouter()


@router.post("/get-answer-langraph-hybrid")
def get_answer_langraph_hybrid_endpoint(req: AnswerLanggraphRequest):
    """
    Get an answer using the LangGraph hybrid retrieval pipeline.
    
    - If generator_prompt_id and validator_prompt_id are provided, uses those specific prompts
    - Otherwise, if user_id is provided, uses that user's active generator and validator prompts
    - Otherwise, uses system default prompts

    """
    return answer_langgraph_hybrid(
        question=req.question,
        file_filter=req.file_filter,
        generator_prompt_id=req.generator_prompt_id,
        validator_prompt_id=req.validator_prompt_id,
        user_id=req.user_id
    )