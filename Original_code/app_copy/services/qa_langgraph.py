from __future__ import annotations
import logging
import os
import time
from typing import Optional
import json

from langchain_openai import ChatOpenAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langgraph.graph import StateGraph, END

from app.services.hybrid import hybrid_retrieve
from app.services.llm_gateway import get_access_token
from app.services.prompts import get_active_prompt_by_type, get_prompt_by_id
from app.core.config import config
from app.models.schemas import GeneratorOutput,ValidatorOutput
logger = logging.getLogger(__name__)

def _get_llm() -> ChatOpenAI:
    """Create an LLM client configured for the LLM gateway."""

    access_token = get_access_token()

    return ChatOpenAI(
        model=os.getenv("CHAT_MODEL", "gpt-4.1"),
        openai_api_key=config.llm_gateway_key,
        openai_api_base=config.base_url,
        default_headers={
            "Authorization": f"Bearer {access_token}",
            "X-LLM-Gateway-Key": config.llm_gateway_key,
        },
        streaming=True
        # callbacks=[StreamingStdOutCallbackHandler()],
    )


def _get_prompt_template(prompt_type: str, prompt_id: int | None = None, user_id: str | None = None) -> str:
    """Get prompt template by ID or by fetching the active prompt for a user or system default."""
    if prompt_id:
        prompt = get_prompt_by_id(prompt_id)
        if not prompt:
            raise ValueError(f"Prompt with id={prompt_id} not found")
        if prompt['type'] != prompt_type:
            raise ValueError(f"Prompt with id={prompt_id} is type '{prompt['type']}', expected '{prompt_type}'")
    else:
        prompt = get_active_prompt_by_type(prompt_type, user_id)
        if not prompt:
            raise ValueError(f"No active {prompt_type} prompt found")
    return prompt['template']


def _generator_agent(
    llm: ChatOpenAI,
    *,
    question: str,
    context: str,
    generator_prompt_id: int | None = None,
    user_id: str | None = None
) -> str:
    logger.info(f"Starting generator agent")
    template = _get_prompt_template('generator', generator_prompt_id, user_id)
    try:
        prompt = template.format(context=context, question=question)
    except Exception as e:
        # If user-provided template is invalid, fall back to a simple prompt.
        logger.warning(f"Failed to format generator template: {type(e).__name__}")
 
    # Configure LLM with structured output
    generator_structured_llm = llm.with_structured_output(GeneratorOutput)

    try:
        generator_response = generator_structured_llm.invoke(prompt)
        logger.info(f"Generator agent completed successfully")
        return generator_response.model_dump()
    except Exception as e:
        logger.error(f"Generator agent LLM call failed: {type(e).__name__}: {str(e)}", exc_info=True)
        raise
 
 
def _validator_agent(
    llm: ChatOpenAI,
    *,
    question: str,
    answer: str,
    context: str,
    validator_prompt_id: int | None = None,
    user_id: str | None = None
) -> dict:
    logger.info(f"Starting validator agent")
    template = _get_prompt_template('validator', validator_prompt_id, user_id)
    try:
        prompt = template.format(context=context, question=question, answer=answer)
    except Exception as e:
        # If user-provided template is invalid.
        logger.warning(f"Failed to format validator template: {type(e).__name__}.")

    # Configure LLM with structured output
    validator_structured_llm = llm.with_structured_output(ValidatorOutput)

    try:
        validator_response = validator_structured_llm.invoke(prompt)
        logger.info(f"Validator agent completed successfully")
        return validator_response.model_dump()
    except Exception as e:
        logger.error(f"Validator agent LLM call failed: {type(e).__name__}: {str(e)}", exc_info=True)
        raise
 
 
def _build_workflow(
    llm: ChatOpenAI,
    generator_prompt_id: int | None = None,
    validator_prompt_id: int | None = None,
    user_id: str | None = None
):
    # Use dict for state to avoid strict TypedDict typing issues.
    def generator_node(state: dict) -> dict:
        generator_output = _generator_agent(
            llm,
            question=state.get("question", ""),
            context=state.get("context", ""),
            generator_prompt_id=generator_prompt_id,
            user_id=user_id
        )
        state["generator_answer"] = generator_output.get("generator_answer", "")
        state["generator_source"] = generator_output.get("generator_source", [])
        return state

    def validator_node(state: dict) -> dict:
        validator_output = _validator_agent(
            llm,
            question=state.get("question", ""),
            answer=state.get("generator_answer", ""),
            context=state.get("context", ""),
            validator_prompt_id=validator_prompt_id,
            user_id=user_id
        )
        state["validated_answer"] = validator_output.get("validator_answer", "")
        state["validated_source"] = validator_output.get("validator_source", [])
        return state

    graph = StateGraph(dict)
    graph.add_node("generator", generator_node)
    graph.add_node("validator", validator_node)
    graph.add_edge("generator", "validator")
    graph.add_edge("validator", END)
    graph.set_entry_point("generator")
    return graph.compile()


def answer_langgraph_hybrid(
    *,
    question: str,
    file_filter: Optional[str] = None,
    generator_prompt_id: int | None = None,
    validator_prompt_id: int | None = None,
    user_id: str | None = None
) -> dict:
    """
    Answer a question using hybrid retrieval and LangGraph pipeline with structured output.
    
    Args:
        question: The question to answer
        file_filter: Optional filter to limit chunks by filename
        generator_prompt_id: Optional specific ID of generator prompt to use
        validator_prompt_id: Optional specific ID of validator prompt to use
        user_id: Optional user ID to use user-specific prompts (fallback if prompt IDs not provided).
                 If prompt IDs are provided, they take precedence.
                 If not provided, uses system default prompts.
    
    Returns:
        Dictionary with response (JSON), raw_answer, retrieval_details, and timings
        The response contains structured output with answer and source information
    """
    logger.info(f"Starting LangGraph workflow. Question: '{question}', File filter: {file_filter}, User ID: {user_id}")
    
    llm = _get_llm()

    workflow = _build_workflow(
        llm,
        generator_prompt_id=generator_prompt_id,
        validator_prompt_id=validator_prompt_id,
        user_id=user_id
    )

    # defaults
    limit = 5
    alpha = 0.6

    timings: dict[str, float] = {}
    start_total = time.time()

    start_retrieval = time.time()
    try:
        retrieval_payload = hybrid_retrieve(query=question, limit=limit, alpha=alpha, file_filter=file_filter)
        timings["retrieval"] = time.time() - start_retrieval
        retrieved_chunks = [
                            {"chunk_text": chunk.get("chunk_text"), "filename": chunk.get("filename")}
                            for chunk in retrieval_payload.get("results", [])
                            if chunk.get("chunk_text")
                            ]
        
        # Format retrieved chunks into a single context string for the LLM, including filenames if available.
        retrieved_context = "\n".join([
            f"[File: {chunk['filename']}]\n{chunk['chunk_text']}" if chunk.get("filename") else chunk["chunk_text"]
            for chunk in retrieved_chunks
        ])
    except Exception as e:
        logger.error(f"LangGraph workflow failed during retrieval: {type(e).__name__}: {str(e)}", exc_info=True)
        raise

    start_generator = time.time()
    try:
        state = {
            "question": question,
            "context": retrieved_context,
            "validated_answer": "",
            "validated_source": "",
            "generator_answer": "",
            "generator_source": ""
        }
        result = workflow.invoke(state)
        timings["generator"] = time.time() - start_generator
        logger.info(f"LangGraph workflow completed successfully")
    except Exception as e:
        logger.error(f"LangGraph workflow failed: {type(e).__name__}: {str(e)}", exc_info=True)
        raise

    timings["total"] = time.time() - start_total

    # Build structured response JSON
    response_dict = {
        "answer": str(result.get("validated_answer", "")).strip(),
        "source": result.get("validated_source"),
        "generator_answer": str(result.get("generator_answer", "")).strip(),
        "generator_source": result.get("generator_source")
    }


    return {
        "response": response_dict,
        "retrieval_details": retrieved_context,
        "timings": timings,
        "file_filter": file_filter
    }