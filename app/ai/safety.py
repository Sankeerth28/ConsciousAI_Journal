"""Conservative safety interceptor for self-harm, medical boundaries, and crisis support.

IMPORTANT SAFETY & HEURISTIC LIMITATION NOTICE:
Keyword pattern matching is a conservative, rule-based harm reduction heuristic.
It is NOT an exhaustive suicide or mental health assessment system.
It CANNOT detect every crisis, nuanced distress, metaphorical despair, or non-explicit
intent. This application is an automated self-reflection journal companion and does
not provide clinical, psychiatric, or emergency services.
"""

from __future__ import annotations

import re
from typing import Any

from app.ai.schemas import OutputSafetyCheckResult, SafetyCheckResult

# Crisis patterns (immediate self-harm or suicide ideation)
CRISIS_PATTERNS = [
    r"\b((kill|killing)\s+(myself|me))\b",
    r"\b((end|ending)\s+my\s+life)\b",
    r"\b((take|taking)\s+my\s+(own\s+)?life)\b",
    r"\b(commit\s+suicide)\b",
    r"\b(want\s+to\s+die)\b",
    r"\b(suicidal\s+(thoughts|ideation|feelings))\b",
    r"\b(suicide)\b",
    r"\b((harm|harming|hurt|hurting)\s+(myself|me))\b",
    r"\b(self-?harm)\b",
    r"\b(better\s+off\s+dead)\b",
    r"\b(don'?t\s+want\s+to\s+live\s+anymore)\b",
]

# Medical diagnosis / prescription patterns
MEDICAL_PATTERNS = [
    r"\b(diagnose\s+me)\b",
    r"\b(do\s+i\s+have\s+(depression|bipolar|adhd|ptsd|schizophrenia|anxiety\s+disorder|ocd))\b",
    r"\b(prescribe\s+(me\s+)?(medication|drugs|pills|antidepressants))\b",
    r"\b(what\s+dose\s+of\s+\w+\s+should\s+i\s+take)\b",
]

# Post-generation Output Safety Patterns (rejects clinical claims, dependencies, etc.)
OUTPUT_SAFETY_PATTERNS = {
    "diagnosis_claim": [
        r"\b(you\s+(have|are\s+suffering\s+from|exhibit|meet\s+the\s+criteria\s+for)\s+([a-z\s]+)?(clinical\s+depression|depression|bipolar|adhd|ptsd|schizophrenia|ocd|borderline|anxiety\s+disorder))\b",
        r"\b(i\s+diagnose\s+you)\b",
        r"\b(my\s+diagnosis\s+is)\b",
        r"\b(diagnostic\s+impression)\b",
    ],
    "medication_instruction": [
        r"\b(take|increase|decrease|stop\s+(taking|using)|adjust\s+(your)?)\s+(your\s+|the\s+)?(\d+\s*mg|medication|medicine|pills|drugs|treatment|antidepressants|ssri|prozac|lexapro|zoloft|xanax|adderall)\b",
        r"\b(dosage\s+(recommendation|guidance|instructions?))\b",
        r"\b(prescribe|prescription\s+for)\b",
        r"\b(start\s+(taking|treatment))\b",
    ],
    "human_or_therapist_claim": [
        r"\b(as\s+(your\s+)?(therapist|doctor|psychiatrist|psychologist|physician|counselor|clinician))\b",
        r"\b(i('?m|\s+am)\s+(a\s+)?(human|licensed\s+(therapist|doctor|counselor|psychologist|psychiatrist|physician)|medical\s+doctor|psychiatrist|physician|therapist))\b",
        r"\b(in\s+our\s+therapy\s+sessions?)\b",
        r"\b(in\s+my\s+clinical\s+practice)\b",
    ],
    "manipulative_or_dependency_language": [
        r"\b(you\s+(only|just)\s+need\s+me)\b",
        r"\b(don'?t\s+talk\s+to\s+(anyone\s+else|your\s+family|friends|doctor|therapist))\b",
        r"\b(i('?m|\s+am)\s+the\s+only\s+one\s+who\s+(understands|cares\s+about)\s+you)\b",
        r"\b(only\s+trust\s+me)\b",
        r"\b(trust\s+only\s+me)\b",
        r"\b(you\s+can\s+only\s+rely\s+on\s+me)\b",
    ],
    "unsafe_encouragement": [
        r"\b(go\s+ahead\s+and\s+(harm|cut|kill|punish|hurt)\s+(yourself|myself))\b",
        r"\b(give\s+up\s+on\s+life)\b",
        r"\b(ending\s+it\s+all\s+is\s+the\s+best)\b",
        r"\b(you\s+should\s+(harm|hurt|kill)\s+yourself)\b",
    ],
    "overconfident_mental_health_conclusions": [
        r"\b(this\s+(definitely\s+)?proves\s+you\s+(are|have)\s+(mentally\s+ill|depressed|sick|broken))\b",
        r"\b(you\s+are\s+(undeniably|certainly|definitely)\s+(depressed|bipolar|disordered|mentally\s+ill))\b",
        r"\b(there\s+is\s+no\s+doubt\s+you\s+have\s+(depression|bipolar|anxiety|ptsd))\b",
    ],
}

# Region-aware crisis resource database (default resources)
REGION_CRISIS_RESOURCES: dict[str, dict[str, Any]] = {
    "US": {
        "emergency": "911",
        "lifeline": "Call or text 988 (Suicide & Crisis Lifeline)",
        "text": "Text HOME to 741741 (Crisis Text Line)",
        "additional": "Veterans Crisis Line: Dial 988, then press 1",
    },
    "CA": {
        "emergency": "911",
        "lifeline": "Call or text 988 (Suicide Crisis Helpline)",
        "text": "Text 686868 (Youth) or 741741 (Adults)",
        "additional": "Hope for Wellness (Indigenous): 1-855-242-3310",
    },
    "UK": {
        "emergency": "999",
        "lifeline": "Call 111 (NHS Mental Health Services) or 0800 689 5652 (National Suicide Prevention Helpline)",
        "text": "Text SHOUT to 85258",
        "additional": "Samaritans: Call 116 123",
    },
    "AU": {
        "emergency": "000",
        "lifeline": "Call 13 11 14 (Lifeline Australia)",
        "text": "Text 0477 13 11 14",
        "additional": "Suicide Call Back Service: 1300 659 467",
    },
    "IN": {
        "emergency": "112",
        "lifeline": "Call 14416 or 1800-891-4416 (Tele-MANAS)",
        "text": "Kiran Helpline: 1800-599-0019",
        "additional": "Vandrevala Foundation: +91 9999 666 555",
    },
    "GLOBAL": {
        "emergency": "your local emergency number",
        "lifeline": "Find immediate local support at https://findahelpline.com or https://befrienders.org",
        "text": "Reach out to local emergency services or mental health crisis lines",
        "additional": "Befrienders Worldwide: https://www.befrienders.org",
    },
}

MEDICAL_BOUNDARY_RESPONSE = (
    "I hear that you are seeking clarity on your health and well-being. As an AI journaling tool, "
    "I cannot evaluate symptoms, diagnose medical conditions, or recommend medications. "
    "For an accurate assessment and compassionate personalized support, please speak with a "
    "licensed mental health professional or primary care physician."
)


def format_crisis_response(
    region: str = "GLOBAL",
    resources_dict: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Format crisis response with region-aware emergency and trusted-person guidance.

    CRITICAL NOTICE: Keyword pattern matching is a heuristic harm-reduction tool and
    cannot detect every crisis. Automated responses provide emergency numbers and encouragement
    to connect with human support.
    """
    reg_key = (region or "GLOBAL").strip().upper()
    active_resources = resources_dict or REGION_CRISIS_RESOURCES
    resources = active_resources.get(
        reg_key, active_resources.get("GLOBAL", REGION_CRISIS_RESOURCES["GLOBAL"])
    )
    emergency_num = resources.get("emergency", "your local emergency number")
    lifeline_info = resources.get("lifeline", "Find immediate support at https://findahelpline.com")
    text_info = resources.get("text", "Reach out to local emergency services or crisis lines")
    additional_info = resources.get("additional", "")

    msg = (
        "It sounds like you are going through a profoundly difficult moment right now, "
        "and your safety is the utmost priority. While I am an AI reflection companion and cannot "
        "provide crisis intervention or therapy, please know that you do not have to navigate this alone.\n\n"
        "Immediate support is available free, confidential, and 24/7:\n"
        f"• Emergency Services: If you are in immediate danger, call {emergency_num} right now.\n"
        f"• Crisis Lifeline: {lifeline_info}\n"
        f"• Crisis Text Support: {text_info}\n"
    )
    if additional_info:
        msg += f"• Additional Support: {additional_info}\n"

    msg += (
        "\n• Trusted Person Guidance: Please consider reaching out to a family member, close friend, "
        "physician, counselor, or someone you trust. Having someone near you makes a difference.\n"
        "• International Directory: If you are outside this area, find immediate support at https://findahelpline.com\n\n"
        "Please take care of yourself and connect with real human care today."
    )
    return msg


class SafetyInterceptor:
    """Evaluates user input and generated outputs for safety, crises, and boundaries.

    HEURISTIC LIMITATION NOTICE:
    This interceptor uses regex pattern matching as a harm-reduction heuristic. It is NOT
    an exhaustive clinical or psychological assessment and cannot detect every crisis.
    """

    def __init__(
        self,
        crisis_patterns: list[str] | None = None,
        medical_patterns: list[str] | None = None,
        default_region: str = "GLOBAL",
        custom_crisis_resources: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._crisis_regexes = [
            re.compile(p, re.IGNORECASE) for p in (crisis_patterns or CRISIS_PATTERNS)
        ]
        self._medical_regexes = [
            re.compile(p, re.IGNORECASE) for p in (medical_patterns or MEDICAL_PATTERNS)
        ]
        self._default_region = default_region.upper()
        self._crisis_resources = dict(REGION_CRISIS_RESOURCES)
        if custom_crisis_resources:
            self._crisis_resources.update(custom_crisis_resources)

        # Compiled output safety validators
        self._output_regexes: dict[str, list[re.Pattern]] = {
            category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for category, patterns in OUTPUT_SAFETY_PATTERNS.items()
        }

    def check(self, text: str, region: str | None = None) -> SafetyCheckResult:
        """Evaluate input text for safety and return SafetyCheckResult.

        Alias for check_input for backward compatibility.
        """
        return self.check_input(text, region=region)

    def check_input(self, text: str, region: str | None = None) -> SafetyCheckResult:
        """Evaluate user input for crisis ideation or medical boundary violations."""
        if not text or not text.strip():
            return SafetyCheckResult(is_safe=True)

        target_region = (region or self._default_region).upper()

        # 1. Check crisis / self-harm
        for regex in self._crisis_regexes:
            if regex.search(text):
                return SafetyCheckResult(
                    is_safe=False,
                    reason="crisis_detected",
                    crisis_response=format_crisis_response(target_region, self._crisis_resources),
                    region=target_region,
                )

        # 2. Check medical diagnosis requests
        for regex in self._medical_regexes:
            if regex.search(text):
                return SafetyCheckResult(
                    is_safe=False,
                    reason="medical_boundary_violation",
                    crisis_response=MEDICAL_BOUNDARY_RESPONSE,
                    region=target_region,
                )

        return SafetyCheckResult(is_safe=True, region=target_region)

    def check_output(self, text: str) -> OutputSafetyCheckResult:
        """Validate generated reflection text to ensure it adheres to non-clinical boundaries.

        Rejects diagnosis claims, medication/treatment instructions, claims of being a
        human or therapist, manipulative/dependency language, unsafe encouragement,
        and overconfident conclusions.
        """
        if not text or not text.strip():
            return OutputSafetyCheckResult(is_safe=True)

        for category, patterns in self._output_regexes.items():
            for regex in patterns:
                if regex.search(text):
                    return OutputSafetyCheckResult(
                        is_safe=False,
                        violation_type=category,
                        reason=f"Generated output violated safety policy: {category}",
                    )

        return OutputSafetyCheckResult(is_safe=True)
