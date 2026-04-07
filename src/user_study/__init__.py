"""User study: pseudo-heatmap, LLM prompts, export helpers."""

from .pseudo_heatmap import (
    generate_fastcav_pseudo_heatmap,
    save_pseudo_heatmap_bundle,
    scalar_stress_from_concepts,
)
from .heatmap_summary import summarize_heatmap_regions
from .llm_user_study import (
    SYSTEM_PROMPT_STRICT,
    build_explanation_prompt,
    build_fastcav_method_block,
    build_gradcam_method_block,
    build_strict_user_study_user_message,
    format_concept_signals_with_trends,
    format_performance_signals_block,
    generate_explanation_openai,
    narrate_fastcav_visualization_summary,
    narrate_gradcam_visualization_summary,
    write_evaluation_questionnaire,
    write_full_prompt_for_disk,
)

__all__ = [
    "generate_fastcav_pseudo_heatmap",
    "save_pseudo_heatmap_bundle",
    "scalar_stress_from_concepts",
    "summarize_heatmap_regions",
    "SYSTEM_PROMPT_STRICT",
    "build_strict_user_study_user_message",
    "format_concept_signals_with_trends",
    "format_performance_signals_block",
    "narrate_gradcam_visualization_summary",
    "narrate_fastcav_visualization_summary",
    "write_full_prompt_for_disk",
    "build_explanation_prompt",
    "build_fastcav_method_block",
    "build_gradcam_method_block",
    "generate_explanation_openai",
    "write_evaluation_questionnaire",
]
