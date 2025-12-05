"""
DSPy-based Retrieval Prompt Optimizer

A novel approach to optimize information extraction prompts without ground truth data
using synthetic evaluation, self-consistency validation, and multi-model consensus.
"""

import logging
import json
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
import statistics

import dspy
from dspy import Signature, InputField, OutputField
from dspy.teleprompt import MIPROv2, COPRO, BootstrapFewShot
from dspy.evaluate import Evaluate
from dspy.propose import GroundedProposer


logger = logging.getLogger(__name__)


# ============================================================================
# DSPy Signatures for Aspect Harvesting and Analysis
# ============================================================================

class AspectHarvesting(Signature):
    """Extract structured information aspects from email content with high recall."""

    email_metadata = InputField(desc="Email metadata including sender, date, thread info")
    email_body = InputField(desc="Full email body content")

    aspects = OutputField(
        desc="List of extracted aspects with scope_key, anchors, evidence_span, and window_index",
        format="List[Dict] with keys: scope_key, anchors, evidence_span, window_index"
    )


class AspectAnalysis(Signature):
    """Convert harvested aspects into structured, normalized memory propositions."""

    email_metadata = InputField(desc="Email metadata for temporal context")
    email_body = InputField(desc="Original email content for validation")
    aspect = InputField(desc="Single aspect to analyze and potentially promote")

    promoted = OutputField(desc="Boolean indicating if aspect was promoted to memory note")
    notes = OutputField(
        desc="List of structured memory notes if promoted",
        format="List[Dict] with keys: proposition, temporal_context, entities, insight_type, summary_context"
    )
    parked_reason = OutputField(desc="Reason for parking if not promoted (optional)")


class SyntheticEmailGeneration(Signature):
    """Generate realistic financial advisor-client emails for testing."""

    scenario_type = InputField(desc="Type of financial scenario (portfolio_update, meeting_request, document_review, etc.)")
    complexity_level = InputField(desc="Complexity level: simple, medium, complex")
    entity_count = InputField(desc="Number of entities to include (people, accounts, documents)")

    email_content = OutputField(desc="Generated email with realistic financial content")
    expected_aspects = OutputField(desc="Expected aspects that should be extracted from this email")


class QualityValidation(Signature):
    """Validate the quality of extracted aspects and propositions."""

    original_email = InputField(desc="Original email content")
    extracted_aspects = InputField(desc="Aspects extracted by the system")

    quality_score = OutputField(desc="Quality score from 0.0 to 1.0")
    missing_information = OutputField(desc="Important information that was missed")
    hallucinations = OutputField(desc="Information that was hallucinated or incorrect")
    improvements = OutputField(desc="Specific suggestions for improvement")


class ConsistencyValidation(Signature):
    """Check consistency between multiple runs of the same extraction."""

    email_content = InputField(desc="Original email content")
    extraction_run_1 = InputField(desc="First extraction result")
    extraction_run_2 = InputField(desc="Second extraction result")
    extraction_run_3 = InputField(desc="Third extraction result")

    consistency_score = OutputField(desc="Consistency score from 0.0 to 1.0")
    consensus_extraction = OutputField(desc="Best consensus extraction combining all runs")
    inconsistencies = OutputField(desc="Specific inconsistencies found between runs")


# ============================================================================
# Synthetic Evaluation Strategy
# ============================================================================

class SyntheticEvaluationEngine:
    """Generate synthetic data and evaluation metrics without ground truth."""

    def __init__(self, lm_primary=None, lm_validator=None):
        self.lm_primary = lm_primary or dspy.settings.lm
        self.lm_validator = lm_validator or dspy.settings.lm

        # Initialize DSPy modules
        self.email_generator = dspy.Predict(SyntheticEmailGeneration)
        self.quality_validator = dspy.Predict(QualityValidation)
        self.consistency_validator = dspy.Predict(ConsistencyValidation)

    def generate_synthetic_dataset(self, size: int = 100) -> List[Dict[str, Any]]:
        """Generate synthetic financial emails with expected extractions."""

        scenarios = ["portfolio_update", "meeting_request", "document_review",
                    "compliance_notice", "investment_opportunity", "market_update"]
        complexity_levels = ["simple", "medium", "complex"]

        dataset = []
        for i in range(size):
            scenario = scenarios[i % len(scenarios)]
            complexity = complexity_levels[i % len(complexity_levels)]
            entity_count = 2 + (i % 4)  # 2-5 entities

            with dspy.context(lm=self.lm_primary):
                result = self.email_generator(
                    scenario_type=scenario,
                    complexity_level=complexity,
                    entity_count=entity_count
                )

            dataset.append({
                "email_content": result.email_content,
                "expected_aspects": result.expected_aspects,
                "scenario": scenario,
                "complexity": complexity,
                "synthetic_id": f"synthetic_{i:03d}"
            })

        return dataset

    def self_consistency_metric(self, extraction_fn, email_data: Dict, runs: int = 3) -> float:
        """Measure consistency across multiple extraction runs."""

        results = []
        for _ in range(runs):
            # Add slight temperature variation to test consistency
            with dspy.context(lm=self.lm_primary, temperature=0.3):
                result = extraction_fn(email_data)
                results.append(result)

        # Validate consistency using DSPy
        if len(results) >= 3:
            consistency_result = self.consistency_validator(
                email_content=email_data.get("body", ""),
                extraction_run_1=json.dumps(results[0] if results[0] else {}),
                extraction_run_2=json.dumps(results[1] if results[1] else {}),
                extraction_run_3=json.dumps(results[2] if results[2] else {})
            )
            return float(consistency_result.consistency_score)

        return 0.0

    def quality_metric(self, extraction_result: Dict, email_data: Dict) -> float:
        """Evaluate extraction quality using LLM-based validation."""

        with dspy.context(lm=self.lm_validator):
            quality_result = self.quality_validator(
                original_email=email_data.get("body", ""),
                extracted_aspects=json.dumps(extraction_result)
            )
            return float(quality_result.quality_score)


# ============================================================================
# Multi-Stage Prompt Optimization
# ============================================================================

class RetrievalPromptOptimizer:
    """Multi-stage optimizer for retrieval prompts using DSPy techniques."""

    def __init__(self, primary_lm=None, validator_lm=None):
        self.primary_lm = primary_lm or dspy.settings.lm
        self.validator_lm = validator_lm or dspy.settings.lm

        self.synthetic_engine = SyntheticEvaluationEngine(primary_lm, validator_lm)

        # Create the modules we'll optimize
        self.aspect_harvester = dspy.Predict(AspectHarvesting)
        self.aspect_analyzer = dspy.Predict(AspectAnalysis)

    def create_composite_metric(self, weight_consistency=0.4, weight_quality=0.6):
        """Create a composite metric combining consistency and quality."""

        def composite_metric(prediction, example):
            try:
                # Extract the result from prediction
                if hasattr(prediction, 'aspects'):
                    extraction_result = {"aspects": prediction.aspects}
                elif hasattr(prediction, 'notes'):
                    extraction_result = {"notes": prediction.notes}
                else:
                    return 0.0

                email_data = {"body": example.get("email_body", "")}

                # Self-consistency score
                consistency_score = self.synthetic_engine.self_consistency_metric(
                    lambda data: extraction_result, email_data
                )

                # Quality score
                quality_score = self.synthetic_engine.quality_metric(
                    extraction_result, email_data
                )

                # Composite score
                final_score = (weight_consistency * consistency_score +
                             weight_quality * quality_score)

                return final_score

            except Exception as e:
                logger.warning(f"Metric evaluation failed: {e}")
                return 0.0

        return composite_metric

    def optimize_aspect_harvesting(self, synthetic_dataset: List[Dict]) -> dspy.Module:
        """Optimize the aspect harvesting stage using COPRO."""

        # Convert synthetic dataset to DSPy examples
        train_examples = []
        for item in synthetic_dataset[:50]:  # Use first 50 for training
            example = dspy.Example(
                email_metadata={"sent_at": "2025-01-15", "from": "client@example.com"},
                email_body=item["email_content"]
            )
            train_examples.append(example)

        # Use COPRO optimizer
        metric = self.create_composite_metric()
        copro = COPRO(
            metric=metric,
            breadth=8,
            depth=3,
            init_temperature=1.2,
            prompt_model=self.validator_lm
        )

        with dspy.context(lm=self.primary_lm):
            optimized_harvester = copro.compile(
                self.aspect_harvester,
                trainset=train_examples[:30],
                valset=train_examples[30:50]
            )

        return optimized_harvester

    def optimize_aspect_analysis(self, synthetic_dataset: List[Dict]) -> dspy.Module:
        """Optimize the aspect analysis stage using MIPROv2."""

        # Convert to DSPy examples for analysis stage
        train_examples = []
        for item in synthetic_dataset[50:]:  # Use second half for analysis training
            example = dspy.Example(
                email_metadata={"sent_at": "2025-01-15"},
                email_body=item["email_content"],
                aspect={
                    "scope_key": {"entities": ["test"], "attribute": "test", "qualifier": ""},
                    "anchors": {"numbers": [], "dates": [], "verbs": []},
                    "evidence_span": "test span",
                    "window_index": 0
                }
            )
            train_examples.append(example)

        # Use MIPROv2 optimizer
        metric = self.create_composite_metric()
        mipro = MIPROv2(
            metric=metric,
            num_candidates=6,
            init_temperature=1.0
        )

        with dspy.context(lm=self.primary_lm):
            optimized_analyzer = mipro.compile(
                self.aspect_analyzer,
                trainset=train_examples[:20]
            )

        return optimized_analyzer

    def bootstrap_few_shot_examples(self, synthetic_dataset: List[Dict]) -> dspy.Module:
        """Use bootstrap learning to create better few-shot examples."""

        train_examples = []
        for item in synthetic_dataset:
            example = dspy.Example(
                email_metadata={"sent_at": "2025-01-15"},
                email_body=item["email_content"]
            )
            train_examples.append(example)

        # Bootstrap few-shot learning
        bootstrap = BootstrapFewShot(
            metric=self.create_composite_metric(0.3, 0.7),  # Emphasize quality
            max_bootstrapped_demos=4,
            max_labeled_demos=8
        )

        with dspy.context(lm=self.primary_lm):
            bootstrapped_harvester = bootstrap.compile(
                self.aspect_harvester,
                trainset=train_examples
            )

        return bootstrapped_harvester

    def run_full_optimization(self) -> Dict[str, Any]:
        """Run the complete optimization pipeline."""

        logger.info("Starting retrieval prompt optimization...")

        # Step 1: Generate synthetic evaluation data
        logger.info("Generating synthetic dataset...")
        synthetic_dataset = self.synthetic_engine.generate_synthetic_dataset(size=100)

        # Step 2: Optimize aspect harvesting
        logger.info("Optimizing aspect harvesting...")
        optimized_harvester = self.optimize_aspect_harvesting(synthetic_dataset)

        # Step 3: Optimize aspect analysis
        logger.info("Optimizing aspect analysis...")
        optimized_analyzer = self.optimize_aspect_analysis(synthetic_dataset)

        # Step 4: Bootstrap few-shot examples
        logger.info("Bootstrapping few-shot examples...")
        bootstrapped_harvester = self.bootstrap_few_shot_examples(synthetic_dataset)

        # Step 5: Evaluate all variants
        logger.info("Evaluating optimized variants...")
        evaluation_results = self.evaluate_all_variants(
            {
                "original_harvester": self.aspect_harvester,
                "copro_harvester": optimized_harvester,
                "bootstrap_harvester": bootstrapped_harvester,
                "original_analyzer": self.aspect_analyzer,
                "mipro_analyzer": optimized_analyzer
            },
            synthetic_dataset[-20:]  # Use final 20 for testing
        )

        return {
            "optimized_harvester": optimized_harvester,
            "optimized_analyzer": optimized_analyzer,
            "bootstrapped_harvester": bootstrapped_harvester,
            "evaluation_results": evaluation_results,
            "synthetic_dataset_size": len(synthetic_dataset)
        }

    def evaluate_all_variants(self, modules: Dict[str, dspy.Module], test_dataset: List[Dict]) -> Dict[str, float]:
        """Evaluate all module variants on test data."""

        results = {}

        for module_name, module in modules.items():
            scores = []

            for test_item in test_dataset:
                try:
                    with dspy.context(lm=self.primary_lm):
                        if "harvester" in module_name:
                            prediction = module(
                                email_metadata={"sent_at": "2025-01-15"},
                                email_body=test_item["email_content"]
                            )
                        else:  # analyzer
                            prediction = module(
                                email_metadata={"sent_at": "2025-01-15"},
                                email_body=test_item["email_content"],
                                aspect={"scope_key": {}, "anchors": {}, "evidence_span": "", "window_index": 0}
                            )

                        # Score using composite metric
                        example = {"email_body": test_item["email_content"]}
                        score = self.create_composite_metric()(prediction, example)
                        scores.append(score)

                except Exception as e:
                    logger.warning(f"Evaluation failed for {module_name}: {e}")
                    scores.append(0.0)

            results[module_name] = statistics.mean(scores) if scores else 0.0

        return results


# ============================================================================
# Assertion-Based Validation Framework
# ============================================================================

class AssertionValidator:
    """Validation framework using domain-specific assertions."""

    @staticmethod
    def validate_aspect_quality(aspect: Dict) -> Tuple[bool, List[str]]:
        """Validate harvested aspect meets quality requirements."""
        errors = []

        # Must have evidence span
        if not aspect.get("evidence_span"):
            errors.append("Missing evidence span")

        # Evidence span should be reasonable length
        evidence = aspect.get("evidence_span", "")
        if len(evidence) > 280:
            errors.append(f"Evidence span too long: {len(evidence)} chars")

        # Must have at least one anchor
        anchors = aspect.get("anchors", {})
        has_anchor = any(anchors.get(key) for key in anchors.keys())
        if not has_anchor:
            errors.append("No anchors found")

        # Scope key must have entities
        scope_key = aspect.get("scope_key", {})
        if not scope_key.get("entities"):
            errors.append("No entities in scope key")

        return len(errors) == 0, errors

    @staticmethod
    def validate_proposition_quality(note: Dict) -> Tuple[bool, List[str]]:
        """Validate analyzed proposition meets quality requirements."""
        errors = []

        # Must have proposition text
        proposition = note.get("proposition", {})
        if not proposition.get("text"):
            errors.append("Missing proposition text")

        # Proposition should be self-contained
        prop_text = proposition.get("text", "")
        if any(pronoun in prop_text.lower() for pronoun in ["he", "she", "it", "they", "this", "that"]):
            errors.append("Proposition contains pronouns")

        # Must have entities
        if not note.get("entities"):
            errors.append("Missing entities")

        # Must have temporal context
        if not note.get("temporal_context", {}).get("as_of_date"):
            errors.append("Missing temporal context")

        return len(errors) == 0, errors


# ============================================================================
# Usage Example
# ============================================================================

def example_usage():
    """Example of how to use the optimization framework."""

    # Configure DSPy with your models
    primary_model = dspy.OpenAI(model="gpt-3.5-turbo", temperature=0.3)
    validator_model = dspy.OpenAI(model="gpt-4", temperature=0.1)

    dspy.settings.configure(lm=primary_model)

    # Create optimizer
    optimizer = RetrievalPromptOptimizer(
        primary_lm=primary_model,
        validator_lm=validator_model
    )

    # Run optimization
    results = optimizer.run_full_optimization()

    print("Optimization Results:")
    print(f"Generated {results['synthetic_dataset_size']} synthetic examples")
    print("\nModule Performance:")
    for module_name, score in results['evaluation_results'].items():
        print(f"  {module_name}: {score:.3f}")

    return results


if __name__ == "__main__":
    example_usage()